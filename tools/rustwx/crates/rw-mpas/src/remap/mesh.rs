//! One side of a mesh-to-mesh remap: an MPAS mesh with its Voronoi polygons.
//!
//! Everything the remap needs to know about a mesh, read once: cell centres,
//! the CCW vertex ring of every cell (its Voronoi polygon on the sphere),
//! edge midpoints and normals, the dual triangulation for the barycentric
//! fallback, and the lengths the kinetic-energy budget uses.
//!
//! ## Where the polygons come from
//! `verticesOnCell` and `latVertex`/`lonVertex`, exactly as the file stores
//! them.  The polygons are not re-derived from the cell centres: the file's
//! own vertex ring is the region the file's `areaCell` integrates over, so a
//! budget computed from these polygons and one computed from `areaCell`
//! describe the same column.  A cell whose ring names a stored zero (the
//! outer ring of some culled meshes) has no closed polygon; it is marked and
//! the remap places it by the barycentric fallback instead, and the receipt
//! counts it.

#![allow(clippy::needless_range_loop)]

use std::path::Path;

use crate::error::{MpasError, MpasResult};
use crate::mesh::derive::MpasMesh;
use crate::mesh::geom::{self, V3};

/// Mean Earth radius MPAS-A uses (`mpas_constants.F`, `a`).
pub const EARTH_RADIUS_M: f64 = 6_371_229.0;

/// A mesh as the remap sees it.  Connectivity is held 1-based with `0` for a
/// missing entry, exactly as the file stores it, so the barycentric sampler
/// in [`crate::lbc::sphere`] can be built from the same arrays.
#[derive(Debug, Clone)]
pub struct RemapMesh {
    pub n_cells: usize,
    pub n_edges: usize,
    pub max_edges: usize,
    pub vertex_degree: usize,
    /// Metres per unit-sphere radian: the file's `sphere_radius` when it is a
    /// real radius, else Earth's.
    pub radius_m: f64,
    pub cell_lat: Vec<f64>,
    pub cell_lon: Vec<f64>,
    pub cell_xyz: Vec<V3>,
    pub edge_lat: Vec<f64>,
    pub edge_lon: Vec<f64>,
    pub edge_xyz: Vec<V3>,
    pub angle_edge: Vec<f64>,
    /// 3-D unit normal at each edge midpoint, pointing from `cellsOnEdge[0]`
    /// to `cellsOnEdge[1]` (the MPAS convention the sign of `u` is read in).
    pub edge_normal: Vec<V3>,
    pub n_edges_on_cell: Vec<usize>,
    /// `[cell][slot]`, 1-based, 0 = missing.
    pub edges_on_cell: Vec<Vec<usize>>,
    pub cells_on_cell: Vec<Vec<usize>>,
    /// `[edge] = [c1, c2]`, 1-based, 0 = missing.
    pub cells_on_edge: Vec<[usize; 2]>,
    /// `nVertices * vertexDegree`, 1-based, 0 = missing.
    pub cells_on_vertex: Vec<i64>,
    /// The CCW Voronoi polygon of every cell on the unit sphere, or `None`
    /// when its vertex ring is incomplete.
    pub polygons: Vec<Option<Vec<V3>>>,
    /// Geometric polygon area on the unit sphere; for a polygon-less cell the
    /// file's `areaCell` scaled to the unit sphere.
    pub polygon_area: Vec<f64>,
    /// `dcEdge` and `dvEdge` in metres.
    pub dc_edge_m: Vec<f64>,
    pub dv_edge_m: Vec<f64>,
    /// `coeffs_reconstruct` when the file carries a non-zero one:
    /// `[cell][slot] = [x, y, z]`.
    pub coeffs_reconstruct: Option<Vec<Vec<V3>>>,
}

impl RemapMesh {
    /// Cells whose polygon could not be closed.
    pub fn open_polygons(&self) -> usize {
        self.polygons.iter().filter(|p| p.is_none()).count()
    }

    /// Total geometric area in square metres.
    pub fn total_area_m2(&self) -> f64 {
        self.polygon_area.iter().sum::<f64>() * self.radius_m * self.radius_m
    }

    /// Cell area in square metres.
    #[inline]
    pub fn area_m2(&self, c: usize) -> f64 {
        self.polygon_area[c] * self.radius_m * self.radius_m
    }

    /// The valid 0-based cells either side of an edge.
    pub fn edge_cells(&self, e: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(2);
        for &c in &self.cells_on_edge[e] {
            if c >= 1 && c <= self.n_cells {
                out.push(c - 1);
            }
        }
        out
    }

    /// Build a sampler over this mesh's dual triangulation.
    pub fn sampler(&self) -> MpasResult<crate::lbc::sphere::SphereSampler> {
        crate::lbc::sphere::SphereSampler::build(
            &self.cell_lat,
            &self.cell_lon,
            &self.cells_on_vertex,
            self.vertex_degree,
            &self.edge_lat,
            &self.edge_lon,
            &self.angle_edge,
            &self.edges_on_cell,
            &self.n_edges_on_cell,
            &self.cells_on_cell,
        )
    }

    /// Read a mesh from a grid file, with a companion (the state or init
    /// file) consulted first for any variable it also carries.
    pub fn read(primary: &Path, companion: Option<&Path>) -> MpasResult<RemapMesh> {
        let reader = PairReader::open(primary, companion)?;
        let n_cells = reader.dim("nCells")?;
        let n_edges = reader.dim("nEdges")?;
        let n_vertices = reader.dim("nVertices")?;
        let max_edges = reader.dim("maxEdges")?;
        let vertex_degree = reader.dim("vertexDegree").unwrap_or(3);

        let cell_lat = reader.f64s("latCell", n_cells)?;
        let cell_lon = reader.f64s("lonCell", n_cells)?;
        let edge_lat = reader.f64s("latEdge", n_edges)?;
        let edge_lon = reader.f64s("lonEdge", n_edges)?;
        let angle_edge = reader.f64s("angleEdge", n_edges)?;
        let vert_lat = reader.f64s("latVertex", n_vertices)?;
        let vert_lon = reader.f64s("lonVertex", n_vertices)?;
        let n_edges_on_cell: Vec<usize> = reader
            .f64s("nEdgesOnCell", n_cells)?
            .into_iter()
            .map(|v| (v.round().max(0.0) as usize).min(max_edges))
            .collect();
        let edges_on_cell = split_index(&reader.f64s("edgesOnCell", n_cells * max_edges)?, max_edges);
        let cells_on_cell = split_index(&reader.f64s("cellsOnCell", n_cells * max_edges)?, max_edges);
        let vertices_on_cell =
            split_index(&reader.f64s("verticesOnCell", n_cells * max_edges)?, max_edges);
        let coe = split_index(&reader.f64s("cellsOnEdge", n_edges * 2)?, 2);
        let cells_on_edge: Vec<[usize; 2]> = coe.iter().map(|p| [p[0], p[1]]).collect();
        let cells_on_vertex: Vec<i64> = reader
            .f64s("cellsOnVertex", n_vertices * vertex_degree)?
            .into_iter()
            .map(|v| v.round() as i64)
            .collect();

        let radius_m = match reader.attr_f64("sphere_radius") {
            Some(r) if r > 1000.0 => r,
            _ => EARTH_RADIUS_M,
        };
        // dcEdge/dvEdge are stored on whatever sphere the file was written
        // on.  A unit-sphere file stores radians.
        let scale_len = |v: Vec<f64>| -> Vec<f64> {
            let unit = v.iter().copied().fold(0.0f64, f64::max) < 10.0;
            if unit {
                v.into_iter().map(|x| x * radius_m).collect()
            } else {
                v
            }
        };
        let dc_edge_m = scale_len(reader.f64s("dcEdge", n_edges)?);
        let dv_edge_m = scale_len(reader.f64s("dvEdge", n_edges)?);
        let area_cell = reader.f64s("areaCell", n_cells)?;
        let area_unit = area_cell.iter().copied().fold(0.0f64, f64::max) < 1.0;

        let vertex_xyz: Vec<V3> = (0..n_vertices)
            .map(|v| geom::from_lat_lon(vert_lat[v], vert_lon[v]))
            .collect();
        let rings: Vec<Vec<i64>> = vertices_on_cell
            .iter()
            .enumerate()
            .map(|(c, ring)| {
                (0..n_edges_on_cell[c])
                    .map(|i| ring[i] as i64 - 1)
                    .collect()
            })
            .collect();

        let coeffs_reconstruct = match reader.f64s_opt("coeffs_reconstruct") {
            Some(flat) if flat.len() == n_cells * max_edges * 3 && flat.iter().any(|v| *v != 0.0) => {
                Some(
                    (0..n_cells)
                        .map(|c| {
                            (0..max_edges)
                                .map(|i| {
                                    let b = (c * max_edges + i) * 3;
                                    [flat[b], flat[b + 1], flat[b + 2]]
                                })
                                .collect()
                        })
                        .collect(),
                )
            }
            _ => None,
        };

        let mut mesh = assemble(
            cell_lat,
            cell_lon,
            edge_lat,
            edge_lon,
            angle_edge,
            &vertex_xyz,
            &rings,
            n_edges_on_cell,
            edges_on_cell,
            cells_on_cell,
            cells_on_edge,
            cells_on_vertex,
            vertex_degree,
            max_edges,
            radius_m,
            dc_edge_m,
            dv_edge_m,
        )?;
        for c in 0..mesh.n_cells {
            if mesh.polygons[c].is_none() {
                let a = area_cell[c];
                mesh.polygon_area[c] = if area_unit { a } else { a / (radius_m * radius_m) };
            }
        }
        mesh.coeffs_reconstruct = coeffs_reconstruct;
        Ok(mesh)
    }

    /// Build from a mesh held in memory (the generator's own type), for tests
    /// and for in-process callers.
    pub fn from_mpas_mesh(m: &MpasMesh) -> MpasResult<RemapMesh> {
        let latlon = |v: &V3| geom::lat_lon(*v);
        let (cell_lat, cell_lon): (Vec<f64>, Vec<f64>) = m.cell_xyz.iter().map(latlon).unzip();
        let (edge_lat, edge_lon): (Vec<f64>, Vec<f64>) = m.edge_xyz.iter().map(latlon).unzip();
        let n_edges_on_cell: Vec<usize> = m.n_edges_on_cell.iter().map(|&v| v as usize).collect();
        let one_based = |flat: &[i32], width: usize| -> Vec<Vec<usize>> {
            flat.chunks(width)
                .map(|ch| ch.iter().map(|&v| if v >= 0 { v as usize + 1 } else { 0 }).collect())
                .collect()
        };
        let edges_on_cell = one_based(&m.edges_on_cell, m.max_edges);
        let cells_on_cell = one_based(&m.cells_on_cell, m.max_edges);
        let cells_on_edge: Vec<[usize; 2]> = one_based(&m.cells_on_edge, 2)
            .into_iter()
            .map(|p| [p[0], p[1]])
            .collect();
        let cells_on_vertex: Vec<i64> = m.cells_on_vertex.iter().map(|&v| v as i64 + 1).collect();
        let rings: Vec<Vec<i64>> = (0..m.n_cells)
            .map(|c| {
                (0..n_edges_on_cell[c])
                    .map(|i| m.vertices_on_cell[c * m.max_edges + i] as i64)
                    .collect()
            })
            .collect();
        assemble(
            cell_lat,
            cell_lon,
            edge_lat,
            edge_lon,
            m.angle_edge.clone(),
            &m.vertex_xyz,
            &rings,
            n_edges_on_cell,
            edges_on_cell,
            cells_on_cell,
            cells_on_edge,
            cells_on_vertex,
            m.vertex_degree,
            m.max_edges,
            EARTH_RADIUS_M,
            m.dc_edge.iter().map(|v| v * EARTH_RADIUS_M).collect(),
            m.dv_edge.iter().map(|v| v * EARTH_RADIUS_M).collect(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    cell_lat: Vec<f64>,
    cell_lon: Vec<f64>,
    edge_lat: Vec<f64>,
    edge_lon: Vec<f64>,
    angle_edge: Vec<f64>,
    vertex_xyz: &[V3],
    rings: &[Vec<i64>],
    n_edges_on_cell: Vec<usize>,
    edges_on_cell: Vec<Vec<usize>>,
    cells_on_cell: Vec<Vec<usize>>,
    cells_on_edge: Vec<[usize; 2]>,
    cells_on_vertex: Vec<i64>,
    vertex_degree: usize,
    max_edges: usize,
    radius_m: f64,
    dc_edge_m: Vec<f64>,
    dv_edge_m: Vec<f64>,
) -> MpasResult<RemapMesh> {
    let n_cells = cell_lat.len();
    let n_edges = edge_lat.len();
    if n_cells < 3 {
        return Err(MpasError::Refusal(format!(
            "the mesh carries {n_cells} cell(s); a remap needs at least one triangle of cells"
        )));
    }
    let cell_xyz: Vec<V3> = (0..n_cells)
        .map(|c| geom::from_lat_lon(cell_lat[c], cell_lon[c]))
        .collect();
    let edge_xyz: Vec<V3> = (0..n_edges)
        .map(|e| geom::from_lat_lon(edge_lat[e], edge_lon[e]))
        .collect();

    let mut polygons: Vec<Option<Vec<V3>>> = Vec::with_capacity(n_cells);
    let mut polygon_area = vec![0.0f64; n_cells];
    for c in 0..n_cells {
        let ring = &rings[c];
        if ring.len() < 3 || ring.iter().any(|&v| v < 0 || v as usize >= vertex_xyz.len()) {
            polygons.push(None);
            continue;
        }
        let mut poly: Vec<V3> = ring.iter().map(|&v| vertex_xyz[v as usize]).collect();
        let mut area = geom::polygon_area(&poly);
        if area < 0.0 {
            poly.reverse();
            area = -area;
        }
        polygon_area[c] = area;
        polygons.push(Some(poly));
    }

    let edge_normal: Vec<V3> = (0..n_edges)
        .map(|e| {
            let [c1, c2] = cells_on_edge[e];
            if c1 >= 1 && c1 <= n_cells && c2 >= 1 && c2 <= n_cells {
                let d = geom::sub(cell_xyz[c2 - 1], cell_xyz[c1 - 1]);
                if let Some(n) = geom::unit(geom::tangent_at(edge_xyz[e], d)) {
                    return n;
                }
            }
            normal_from_angle(edge_xyz[e], angle_edge[e])
        })
        .collect();

    Ok(RemapMesh {
        n_cells,
        n_edges,
        max_edges,
        vertex_degree,
        radius_m,
        cell_lat,
        cell_lon,
        cell_xyz,
        edge_lat,
        edge_lon,
        edge_xyz,
        angle_edge,
        edge_normal,
        n_edges_on_cell,
        edges_on_cell,
        cells_on_cell,
        cells_on_edge,
        cells_on_vertex,
        polygons,
        polygon_area,
        dc_edge_m,
        dv_edge_m,
        coeffs_reconstruct: None,
    })
}

/// The edge normal from `angleEdge`: `(cos a, sin a)` in local (east, north).
pub fn normal_from_angle(at: V3, angle: f64) -> V3 {
    let (east, north) = tangent_basis(at);
    geom::add(geom::scale(east, angle.cos()), geom::scale(north, angle.sin()))
}

/// Local east/north, with a fixed fallback frame exactly at a pole.
pub fn tangent_basis(at: V3) -> (V3, V3) {
    match geom::east_north(at) {
        Some(pair) => pair,
        None => {
            let east = geom::unit(geom::tangent_at(at, [1.0, 0.0, 0.0])).unwrap_or([0.0, 1.0, 0.0]);
            let north = geom::cross(at, east);
            (east, north)
        }
    }
}

fn split_index(flat: &[f64], width: usize) -> Vec<Vec<usize>> {
    flat.chunks(width)
        .map(|ch| {
            ch.iter()
                .map(|v| {
                    let r = v.round();
                    if r > 0.0 {
                        r as usize
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect()
}

/// Reads a variable from the first of two files that carries it.
pub(crate) struct PairReader {
    first: netcrust::File,
    second: Option<netcrust::File>,
    names: String,
}

impl PairReader {
    pub(crate) fn open(first: &Path, second: Option<&Path>) -> MpasResult<PairReader> {
        for p in std::iter::once(first).chain(second) {
            if !p.exists() {
                return Err(MpasError::Refusal(format!("{} does not exist", p.display())));
            }
        }
        let names = match second {
            None => first.display().to_string(),
            Some(s) => format!("{} or {}", s.display(), first.display()),
        };
        Ok(PairReader {
            first: netcrust::File::open(first)?,
            second: second.map(netcrust::File::open).transpose()?,
            names,
        })
    }

    fn files(&self) -> impl Iterator<Item = &netcrust::File> {
        // The companion first: a state file's own copy of a mesh variable
        // describes the state's mesh.
        self.second.iter().chain(std::iter::once(&self.first))
    }

    pub(crate) fn dim(&self, name: &str) -> MpasResult<usize> {
        self.files()
            .find_map(|f| f.dimension(name).map(|d| d.len()))
            .ok_or_else(|| {
                MpasError::Refusal(format!("{} declares no {name} dimension", self.names))
            })
    }

    pub(crate) fn f64s_opt(&self, name: &str) -> Option<Vec<f64>> {
        for f in self.files() {
            if f.variable(name).is_some() {
                if let Ok(a) = f.read_array_f64_first_record_or_all(name) {
                    return Some(a.into_values());
                }
            }
        }
        None
    }

    pub(crate) fn f64s(&self, name: &str, expect: usize) -> MpasResult<Vec<f64>> {
        let v = self.f64s_opt(name).ok_or_else(|| {
            MpasError::Refusal(format!(
                "no readable {name} in {}; the remap needs it to place this mesh",
                self.names
            ))
        })?;
        if v.len() != expect {
            return Err(MpasError::Refusal(format!(
                "{name} in {} holds {} value(s); the mesh's dimensions say {expect}",
                self.names,
                v.len()
            )));
        }
        Ok(v)
    }

    pub(crate) fn attr_f64(&self, name: &str) -> Option<f64> {
        for f in self.files() {
            if let Some(a) = f.attribute(name) {
                if let Some(v) = attr_number(&a) {
                    return Some(v);
                }
            }
        }
        None
    }
}

pub(crate) fn attr_number(a: &netcrust::Attribute) -> Option<f64> {
    a.as_f64()
        .or_else(|| a.as_string().and_then(|t| t.trim().parse().ok()))
}
