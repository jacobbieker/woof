// MPAS-Seaice v8.4.2: incremental remap advection (planar meshes).
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_advection_incremental_remap.F:
//     66-77     module parameters (eps11, the triangle quadrature)
//     3418-3579 make_masks
//     3594-4203 construct_linear_tracer_fields (planar), with
//     4218-4435 compute_gradient_2d, 4450-4672 compute_gradient_3d (planar),
//     4687-4816 compute_barycenter_coordinates,
//     4831-5011 limit_tracer_gradient_2d, 5026-5265 limit_tracer_gradient_3d
//     5280-5375 find_departure_points
//     5390-6280 find_departure_triangles (the planar branches)
//     6295-6522 shift_vertices_of_departure_triangle (the planar branch)
//     6568-6674 get_triangle_quadrature_points (nQuadPoints = 6)
//     8995-9077 find_line_intersection, 9144-9156 cross_product_2d,
//     9198-9246 quadrilateral_area, triangle_area, 9262-9295 point_in_half_plane
//     9309-9387 volume_to_thickness, thickness_to_volume
// (William Lipscomb, LANL, 2015). Copyright (c) 2015, Los Alamos National
// Security, LLC (LANS) and the University Corporation for Atmospheric
// Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// One source for both of MPAS's precisions: RKIND = 4 (the default build) by
// default, RKIND = 8 when MPAS_RKIND8 is defined. Every operation is an
// explicit round-to-nearest intrinsic in the Fortran's evaluation order
// (gfortran -O0 -ffp-contract=off: left to right, parentheses kept, no
// contraction); x**2 is x*x as gfortran expands it. A literal written _RKIND
// is decimal rounded once to RKIND (the f-suffixed literal at RKIND 4).
//
// Layout: a Fortran array a(d1, d2, ..., dn) is C-order (dn, ..., d2, d1); a
// tracer t(nLayers, nCategories, nCells+1) is t[(iCell*nCategories + iCat)*nLayers + iLayer].

#ifdef MPAS_RKIND8
typedef double rk;
#define A(a,b) __dadd_rn((a),(b))
#define S(a,b) __dsub_rn((a),(b))
#define M(a,b) __dmul_rn((a),(b))
#define D(a,b) __ddiv_rn((a),(b))
#define SQRT(a) __dsqrt_rn(a)
#define ABS(a) fabs(a)
#define C(x) x
#define IR_HUGE 1.7976931348623157e308       // huge(0.0_RKIND)
#else
typedef float rk;
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
#define SQRT(a) __fsqrt_rn(a)
#define ABS(a) fabsf(a)
#define C(x) x##f
#define IR_HUGE 3.40282347e38f               // huge(0.0_RKIND)
#endif

// remap.F:66-77
#define IR_EPS11 C(1.0e-11)
#define IR_W1 C(1.09951743655321885e-01)
#define IR_W2 C(2.23381589678011389e-01)
#define IR_Q1 C(9.15762135097710761e-02)
#define IR_Q2 C(8.16847572980458514e-01)
#define IR_Q3 C(1.08103018168070275e-01)
#define IR_Q4 C(4.45948490915965612e-01)

__device__ __forceinline__ int ir_thread() { return blockIdx.x * blockDim.x + threadIdx.x; }

// ---------------------------------------------------------------------------
// volume_to_thickness (9309-9343) and thickness_to_volume (9356-9385) over a
// whole (nCategories, nCells+1) array, as seaice_run_advection_incremental_remap
// calls them on the ice and snow volume.
extern "C" __global__ void seaice_ir_volume_to_thickness_v842(rk *volume, const rk *area, int n)
{
    const int i = ir_thread();
    if (i >= n) return;
    if (area[i] > C(0.0)) {
        volume[i] = D(volume[i], area[i]);
    } else {
        volume[i] = C(0.0);
    }
}

extern "C" __global__ void seaice_ir_thickness_to_volume_v842(rk *thickness, const rk *area, int n)
{
    const int i = ir_thread();
    if (i >= n) return;
    thickness[i] = M(area[i], thickness[i]);
}

// ---------------------------------------------------------------------------
// find_departure_points (5280-5375). The caller zeroes departurePoint(2,
// nVertices+1); for iVertex = 1..nVertices the departure point is the drift
// back over dt, -(u*dt) as gfortran parses -uVelocity*dt, and cfl flags a
// trajectory longer than the shortest edge at the vertex (v8.4.2 logs it and
// carries on).
extern "C" __global__ void seaice_ir_departure_points_v842(
    rk *departurePoint, int *cfl, const rk *uVelocity, const rk *vVelocity, const rk *minLengthEdgesOnVertex,
    rk dt, int nVertices)
{
    const int v = ir_thread();
    if (v >= nVertices) return;
    const rk d1 = -M(uVelocity[v], dt);
    const rk d2 = -M(vVelocity[v], dt);
    departurePoint[2 * v] = d1;
    departurePoint[2 * v + 1] = d2;
    const rk trajectoryLength = SQRT(A(M(d1, d1), M(d2, d2)));
    cfl[v] = trajectoryLength > minLengthEdgesOnVertex[v] ? 1 : 0;
}

// point_in_half_plane(point, lineStart, lineEnd) (9262-9295): the z component of
// (lineEnd - lineStart) x (point - lineStart) is not negative. The remap calls
// it as point_in_half_plane(V1, V2, D), so the arguments read (point, start, end).
__device__ __forceinline__ bool ir_point_in_half_plane(rk px, rk py, rk sx, rk sy, rk ex, rk ey)
{
    const rk v1x = S(ex, sx), v1y = S(ey, sy);
    const rk v2x = S(px, sx), v2y = S(py, sy);
    return S(M(v1x, v2y), M(v1y, v2x)) >= C(0.0);
}

// find_line_intersection (8995-9077): segment p1-p2 against segment p3-p4
__device__ bool ir_line_intersection(rk x1, rk y1, rk x2, rk y2, rk x3, rk y3, rk x4, rk y4, rk *ix, rk *iy)
{
    const rk rx = S(x2, x1), ry = S(y2, y1);
    const rk sx = S(x4, x3), sy = S(y4, y3);
    const rk rsCross = S(M(rx, sy), M(ry, sx));
    const rk rsCrossMin = M(IR_EPS11, SQRT(M(A(M(rx, rx), M(ry, ry)), A(M(sx, sx), M(sy, sy)))));
    bool hit = false;
    if (ABS(rsCross) > rsCrossMin) {
        const rk t1 = D(S(M(sy, S(x3, x1)), M(sx, S(y3, y1))), rsCross);
        const rk t2 = D(S(M(ry, S(x3, x1)), M(rx, S(y3, y1))), rsCross);
        *ix = A(x1, M(t1, rx));
        *iy = A(y1, M(t1, ry));
        if (t1 > C(0.0) && t1 < C(1.0) && t2 > C(0.0) && t2 < C(1.0)) hit = true;
    } else {
        *ix = IR_HUGE;
        *iy = IR_HUGE;
    }
    return hit;
}

// triangle_area (9233-9246)
__device__ __forceinline__ rk ir_triangle_area(rk x1, rk y1, rk x2, rk y2, rk x3, rk y3)
{
    return ABS(M(C(0.5), S(M(S(x2, x1), S(y3, y1)), M(S(y2, y1), S(x3, x1)))));
}

// the vertexOnCell index (1-based) of vertex iVertex in cell iCell, or keep
// (the Fortran loop leaves triangleVertexOnCell as it was without a match)
__device__ __forceinline__ int ir_vertex_on_cell(int iVertex, int iCell, int keep, const int *nEdgesOnCell,
                                                 const int *verticesOnCell, int maxEdges)
{
    int found = keep;
    const int n = nEdgesOnCell[iCell - 1];
    for (int i = 0; i < n; ++i)
        if (iVertex == verticesOnCell[(iCell - 1) * maxEdges + i]) found = i + 1;
    return found;
}

// find_departure_triangles (5390-6280) for one edge, with
// shift_vertices_of_departure_triangle's planar branch (6500-6520). The caller
// zeroes xTriangle, yTriangle, triangleArea, iCellTriangle and maskEdge.
// error[0] is set where v8.4.2 would read a triangle's vertex index it never
// set, or write a seventh triangle (an inconsistent mesh).
extern "C" __global__ void seaice_ir_departure_triangles_v842(
    rk *xTriangle, rk *yTriangle, int *iCellTriangle, rk *triangleArea, int *maskEdge, int *error,
    const rk *departurePointIn, const rk *xVertexOnEdge, const rk *yVertexOnEdge,
    const rk *xVertexOnCell, const rk *yVertexOnCell, const int *nEdgesOnCell, const int *verticesOnCell,
    const int *verticesOnEdge, const int *remapEdge, const int *cellsOnEdgeRemap, const int *edgesOnEdgeRemap,
    int nEdges, int nCells, int maxEdges, int vertexDegree)
{
    const int e = ir_thread();
    if (e >= nEdges) return;
    // the mask (5603-5611)
    int mask = 0;
    if (remapEdge[e] == 1) {
        for (int k = 0; k < 2; ++k) {
            const int v = verticesOnEdge[2 * e + k] - 1;
            const rk d1 = departurePointIn[2 * v], d2 = departurePointIn[2 * v + 1];
            const rk lenSquared = A(M(d1, d1), M(d2, d2));
            if (lenSquared > C(0.0)) mask = 1;
        }
    }
    if (mask == 0) return;
    maskEdge[e] = 1;
#define XVE(i) xVertexOnEdge[e * 8 + (i) - 1]
#define YVE(i) yVertexOnEdge[e * 8 + (i) - 1]
#define CER(i) cellsOnEdgeRemap[e * 6 + (i) - 1]
#define EER(i) edgesOnEdgeRemap[e * 6 + (i) - 1]
    rk dpx[2], dpy[2], tx[6][3], ty[6][3];
    int cellT[6], onEdge[6], onCell[6], sign[6];
    for (int t = 0; t < 6; ++t) { cellT[t] = 0; onEdge[t] = 0; onCell[t] = 0; sign[t] = 0; }
    for (int k = 0; k < 2; ++k) {
        const int v = verticesOnEdge[2 * e + k] - 1;
        dpx[k] = A(XVE(k + 1), departurePointIn[2 * v]);
        dpy[k] = A(YVE(k + 1), departurePointIn[2 * v + 1]);
    }
    const rk evx[2] = {XVE(1), XVE(2)}, evy[2] = {YVE(1), YVE(2)};
    int count = 0;
    bool bad = false;
    // the side edges E1..E4 (5690-5975)
    for (int k = 0; k < 2; ++k) {
        const int vertexK = verticesOnEdge[2 * e + k];
        for (int side = 0; side < 2; ++side) {
            const int iEdgeOnEdgeRemap = k + 1 + 2 * side;
            const int iVertexOnEdgeRemap = iEdgeOnEdgeRemap + 2;
            const int neighbour = EER(iEdgeOnEdgeRemap);
            rk ipx = C(0.0), ipy = C(0.0);
            bool hit = false;
            if (neighbour >= 1 && neighbour <= nEdges)
                hit = ir_line_intersection(dpx[0], dpy[0], dpx[1], dpy[1], evx[k], evy[k],
                                           XVE(iVertexOnEdgeRemap), YVE(iVertexOnEdgeRemap), &ipx, &ipy);
            if (!hit) continue;
            if (count >= 6) { bad = true; break; }
            int t = count++;
            tx[t][0] = evx[k]; ty[t][0] = evy[k];
            tx[t][1] = dpx[k]; ty[t][1] = dpy[k];
            tx[t][2] = ipx; ty[t][2] = ipy;
            onEdge[t] = k + 1;
            cellT[t] = CER(k + 3);
            onCell[t] = ir_vertex_on_cell(vertexK, cellT[t], onCell[t], nEdgesOnCell, verticesOnCell, maxEdges);
            sign[t] = side == 0 ? 1 : -1;
            if (vertexDegree == 4) {
                const int neighbour2 = EER(k + 5);
                rk ipmx = C(0.0), ipmy = C(0.0);
                bool hitMain = false;
                if (neighbour2 >= 1 && neighbour2 <= nEdges)
                    hitMain = ir_line_intersection(dpx[0], dpy[0], dpx[1], dpy[1], evx[k], evy[k],
                                                   XVE(k + 7), YVE(k + 7), &ipmx, &ipmy);
                if (hitMain) {
                    tx[t][2] = ipmx; ty[t][2] = ipmy;
                    if (side == 0) {
                        cellT[t] = CER(k + 5);
                        onCell[t] = ir_vertex_on_cell(vertexK, cellT[t], onCell[t], nEdgesOnCell, verticesOnCell,
                                                      maxEdges);
                    }
                    if (count >= 6) { bad = true; break; }
                    const int t2 = count++;
                    tx[t2][0] = evx[k]; ty[t2][0] = evy[k];
                    tx[t2][1] = ipmx; ty[t2][1] = ipmy;
                    tx[t2][2] = ipx; ty[t2][2] = ipy;
                    cellT[t2] = side == 0 ? CER(k + 3) : CER(k + 5);
                    onEdge[t2] = onEdge[t2 - 1];
                    onCell[t2] = ir_vertex_on_cell(vertexK, cellT[t2], onCell[t2], nEdgesOnCell, verticesOnCell,
                                                   maxEdges);
                    sign[t2] = sign[t2 - 1];
                } else if (side == 1) {
                    cellT[t] = CER(k + 5);
                    onCell[t] = ir_vertex_on_cell(vertexK, cellT[t], onCell[t], nEdgesOnCell, verticesOnCell,
                                                  maxEdges);
                }
            }
            // the departure point moves to the side intersection
            dpx[k] = ipx;
            dpy[k] = ipy;
        }
        if (bad) break;
    }
    // the triangles in C1 and/or C2 (5984-6241)
    rk ipmx = C(0.0), ipmy = C(0.0);
    const bool hitMain = !bad && ir_line_intersection(dpx[0], dpy[0], dpx[1], dpy[1], evx[0], evy[0], evx[1],
                                                     evy[1], &ipmx, &ipmy);
    if (bad) {
    } else if (hitMain) {
        for (int k = 0; k < 2; ++k) {
            if (count >= 6) { bad = true; break; }
            const int t = count++;
            tx[t][0] = evx[k]; ty[t][0] = evy[k];
            tx[t][1] = dpx[k]; ty[t][1] = dpy[k];
            tx[t][2] = ipmx; ty[t][2] = ipmy;
            onEdge[t] = k + 1;
            const bool left = ir_point_in_half_plane(evx[0], evy[0], evx[1], evy[1], dpx[k], dpy[k]);
            cellT[t] = left ? CER(1) : CER(2);
            sign[t] = left ? 1 : -1;
            onCell[t] = ir_vertex_on_cell(verticesOnEdge[2 * e + k], cellT[t], onCell[t], nEdgesOnCell,
                                          verticesOnCell, maxEdges);
        }
    } else {
        const rk quadArea = A(ir_triangle_area(evx[0], evy[0], evx[1], evy[1], dpx[1], dpy[1]),
                              ir_triangle_area(evx[0], evy[0], dpx[1], dpy[1], dpx[0], dpy[0]));
        if (quadArea > C(0.0)) {
            if (count + 2 > 6) {
                bad = true;
            } else {
                int t = count++;
                tx[t][0] = evx[0]; ty[t][0] = evy[0];
                tx[t][1] = evx[1]; ty[t][1] = evy[1];
                tx[t][2] = dpx[0]; ty[t][2] = dpy[0];
                onEdge[t] = 1;
                bool left = ir_point_in_half_plane(evx[0], evy[0], evx[1], evy[1], dpx[0], dpy[0]);
                cellT[t] = left ? CER(1) : CER(2);
                sign[t] = left ? 1 : -1;
                onCell[t] = ir_vertex_on_cell(verticesOnEdge[2 * e], cellT[t], onCell[t], nEdgesOnCell,
                                              verticesOnCell, maxEdges);
                t = count++;
                tx[t][0] = evx[1]; ty[t][0] = evy[1];
                tx[t][1] = dpx[0]; ty[t][1] = dpy[0];
                tx[t][2] = dpx[1]; ty[t][2] = dpy[1];
                onEdge[t] = 2;
                left = ir_point_in_half_plane(evx[0], evy[0], evx[1], evy[1], dpx[1], dpy[1]);
                cellT[t] = left ? CER(1) : CER(2);
                sign[t] = left ? 1 : -1;
                onCell[t] = ir_vertex_on_cell(verticesOnEdge[2 * e + 1], cellT[t], onCell[t], nEdgesOnCell,
                                              verticesOnCell, maxEdges);
            }
        }
    }
    if (bad) { error[0] = 1; return; }
    // each triangle in the frame of its source cell, with its signed area (6247-6276)
    for (int t = 0; t < count; ++t) {
        const int iCell = cellT[t];
        iCellTriangle[e * 6 + t] = iCell;
        if (iCell >= 1 && iCell <= nCells) {
            if (onCell[t] < 1) { error[0] = 1; return; }
            const rk xe = XVE(onEdge[t]), ye = YVE(onEdge[t]);
            const rk xc = xVertexOnCell[(iCell - 1) * maxEdges + onCell[t] - 1];
            const rk yc = yVertexOnCell[(iCell - 1) * maxEdges + onCell[t] - 1];
            for (int j = 0; j < 3; ++j) {
                tx[t][j] = A(S(tx[t][j], xe), xc);
                ty[t][j] = A(S(ty[t][j], ye), yc);
            }
            const rk area = ir_triangle_area(tx[t][0], ty[t][0], tx[t][1], ty[t][1], tx[t][2], ty[t][2]);
            triangleArea[e * 6 + t] = M(area, (rk)sign[t]);
        }
        for (int j = 0; j < 3; ++j) {
            xTriangle[(e * 6 + t) * 6 + j] = tx[t][j];
            yTriangle[(e * 6 + t) * 6 + j] = ty[t][j];
        }
    }
#undef XVE
#undef YVE
#undef CER
#undef EER
}

// get_triangle_quadrature_points (6627-6666): the six points of each triangle
// of each masked edge, from its three vertices
extern "C" __global__ void seaice_ir_quadrature_points_v842(
    rk *xTriangle, rk *yTriangle, const int *maskEdge, int nEdges)
{
    const int i = ir_thread();
    if (i >= nEdges * 6) return;
    const int e = i / 6;
    if (maskEdge[e] != 1) return;
    rk *x = xTriangle + i * 6, *y = yTriangle + i * 6;
    const rk x1 = x[0], y1 = y[0], x2 = x[1], y2 = y[1], x3 = x[2], y3 = y[2];
    x[0] = A(A(M(IR_Q1, x1), M(IR_Q1, x2)), M(IR_Q2, x3));
    y[0] = A(A(M(IR_Q1, y1), M(IR_Q1, y2)), M(IR_Q2, y3));
    x[1] = A(A(M(IR_Q1, x1), M(IR_Q2, x2)), M(IR_Q1, x3));
    y[1] = A(A(M(IR_Q1, y1), M(IR_Q2, y2)), M(IR_Q1, y3));
    x[2] = A(A(M(IR_Q2, x1), M(IR_Q1, x2)), M(IR_Q1, x3));
    y[2] = A(A(M(IR_Q2, y1), M(IR_Q1, y2)), M(IR_Q1, y3));
    x[3] = A(A(M(IR_Q3, x1), M(IR_Q4, x2)), M(IR_Q4, x3));
    y[3] = A(A(M(IR_Q3, y1), M(IR_Q4, y2)), M(IR_Q4, y3));
    x[4] = A(A(M(IR_Q4, x1), M(IR_Q3, x2)), M(IR_Q4, x3));
    y[4] = A(A(M(IR_Q4, y1), M(IR_Q3, y2)), M(IR_Q4, y3));
    x[5] = A(A(M(IR_Q4, x1), M(IR_Q4, x2)), M(IR_Q3, x3));
    y[5] = A(A(M(IR_Q4, y1), M(IR_Q4, y2)), M(IR_Q3, y3));
}

// ---------------------------------------------------------------------------
// make_masks (3418-3579), one tracer. The caller zeroes the tracer's mask (the
// whole array). The mass-like field: every value of cells 1..nCells
// meaningful, and maskCell where its sum over the cell is positive (gfortran's
// SUM: from zero, in array order). A child: meaningful where its parent's
// value exceeds the threshold (the parent's category for a 2-D parent).
extern "C" __global__ void seaice_ir_mask_mass_v842(
    int *mask, int *maskCell, const rk *field, int nCells, int nValues)
{
    const int c = ir_thread();
    if (c >= nCells) return;
    rk massSumCell = C(0.0);
    for (int i = 0; i < nValues; ++i) {
        mask[c * nValues + i] = 1;
        massSumCell = A(massSumCell, field[c * nValues + i]);
    }
    maskCell[c] = massSumCell > C(0.0) ? 1 : 0;
}

extern "C" __global__ void seaice_ir_mask_child_v842(
    int *mask, const rk *parent, rk threshold, int nCells, int nCategories, int nLayers, int parentLayers)
{
    const int i = ir_thread();
    if (i >= nCells * nCategories * nLayers) return;
    const int layer = i % nLayers, cc = i / nLayers;
    const rk p = parent[cc * parentLayers + (parentLayers == 1 ? 0 : layer)];
    if (p > threshold) mask[i] = 1;
}

// SSE maxss/minss with the destination first, as the limiter's MAX/MIN are
// compiled: the ties they can meet (+0 against -0) leave the limited gradient
// the same either way (the factor is max(0, factor - eps11), never a zero
// that a sign could reach)
__device__ __forceinline__ rk ir_max(rk a, rk b) { return a > b ? a : b; }
__device__ __forceinline__ rk ir_min(rk a, rk b) { return a < b ? a : b; }

// compute_barycenter_coordinates (4687-4816): one, two or three fields
// (mass-like, tracer 1, tracer 2); g: the cell's moments x, y, xx, xy, yy,
// xxx, xxy, xyy, yyy, xxxx, xxxy, xxyy, xyyy, yyyy
__device__ void ir_barycenter(int n, const rk *mean, const rk *center, const rk *xg, const rk *yg, const rk *g,
                              rk *xb, rk *yb)
{
    rk reciprocal, prod;
    if (n == 1) {
        const rk c0 = center[0], cx = xg[0], cy = yg[0];
        reciprocal = ABS(mean[0]) > C(0.0) ? D(C(1.0), mean[0]) : C(0.0);
        *xb = M(A(A(M(c0, g[0]), M(cx, g[2])), M(cy, g[3])), reciprocal);
        *yb = M(A(A(M(c0, g[1]), M(cx, g[3])), M(cy, g[4])), reciprocal);
    } else if (n == 2) {
        const rk c0 = M(center[0], center[1]);
        const rk cx = A(M(center[0], xg[1]), M(xg[0], center[1]));
        const rk cy = A(M(center[0], yg[1]), M(yg[0], center[1]));
        const rk cxx = M(xg[0], xg[1]);
        const rk cxy = A(M(xg[0], yg[1]), M(yg[0], xg[1]));
        const rk cyy = M(yg[0], yg[1]);
        prod = M(mean[0], mean[1]);
        reciprocal = ABS(prod) > C(0.0) ? D(C(1.0), prod) : C(0.0);
        *xb = M(A(A(A(A(A(M(c0, g[0]), M(cx, g[2])), M(cy, g[3])), M(cxx, g[5])), M(cxy, g[6])), M(cyy, g[7])),
                reciprocal);
        *yb = M(A(A(A(A(A(M(c0, g[1]), M(cx, g[3])), M(cy, g[4])), M(cxx, g[6])), M(cxy, g[7])), M(cyy, g[8])),
                reciprocal);
    } else {
        const rk c0 = center[0], c1 = center[1], c2 = center[2];
        const rk x0 = xg[0], x1 = xg[1], x2 = xg[2], y0 = yg[0], y1 = yg[1], y2 = yg[2];
        const rk k0 = M(M(c0, c1), c2);
        const rk kx = A(A(M(M(c0, c1), x2), M(M(c0, x1), c2)), M(M(x0, c1), c2));
        const rk ky = A(A(M(M(c0, c1), y2), M(M(c0, y1), c2)), M(M(y0, c1), c2));
        const rk kxx = A(A(M(M(c0, x1), x2), M(M(x0, c1), x2)), M(M(x0, x1), c2));
        const rk kxy = A(A(A(A(A(M(M(c0, x1), y2), M(M(x0, y1), c2)), M(M(y0, c1), x2)), M(M(c0, y1), x2)),
                           M(M(x0, c1), y2)), M(M(y0, x1), c2));
        const rk kyy = A(A(M(M(c0, y1), y2), M(M(y0, c1), y2)), M(M(y0, y1), c2));
        const rk kxxx = M(M(x0, x1), x2);
        const rk kxxy = A(A(M(M(x0, x1), y2), M(M(x0, y1), x2)), M(M(y0, x1), x2));
        const rk kxyy = A(A(M(M(y0, y1), x2), M(M(y0, x1), y2)), M(M(x0, y1), y2));
        const rk kyyy = M(M(y0, y1), y2);
        prod = M(M(mean[0], mean[1]), mean[2]);
        reciprocal = ABS(prod) > C(0.0) ? D(C(1.0), prod) : C(0.0);
        *xb = M(A(A(A(A(A(A(A(A(A(M(k0, g[0]), M(kx, g[2])), M(ky, g[3])), M(kxx, g[5])), M(kxy, g[6])),
                           M(kyy, g[7])), M(kxxx, g[9])), M(kxxy, g[10])), M(kxyy, g[11])), M(kyyy, g[12])),
                reciprocal);
        *yb = M(A(A(A(A(A(A(A(A(A(M(k0, g[1]), M(kx, g[3])), M(ky, g[4])), M(kxx, g[6])), M(kxy, g[7])),
                           M(kyy, g[8])), M(kxxx, g[10])), M(kxxy, g[11])), M(kxyy, g[12])), M(kyyy, g[13])),
                reciprocal);
    }
}

// construct_linear_tracer_fields (3594-4203) for one tracer, one value
// (layer, category) of cells 1..nCells: compute_gradient (planar), the
// limiter against the parent's barycenter (the cell's geometric centre for
// the mass-like field), the centre value, and, for a tracer with children, its
// barycenter (compute_barycenter_coordinates with its ancestors). The caller
// zeroes centre, gradients and barycenters (whole arrays).
//   anc: the parent's and the grandparent's (array, centre, x/y gradient)
//   fields (null when absent); ancLayers their first extents.
extern "C" __global__ void seaice_ir_construct_v842(
    rk *center, rk *xGrad, rk *yGrad, rk *xBarycenter, rk *yBarycenter,
    const rk *field, const int *mask, const int *maskCell,
    const int *nEdgesOnCell, const int *cellsOnCell, const int *edgesOnCell, const int *cellsOnEdge,
    const rk *dcEdge, const rk *coeffsReconstruct, const rk *xVertexOnCell, const rk *yVertexOnCell,
    const rk *g0, const rk *g1, const rk *g2, const rk *g3, const rk *g4, const rk *g5, const rk *g6,
    const rk *g7, const rk *g8, const rk *g9, const rk *g10, const rk *g11, const rk *g12, const rk *g13,
    const rk *parentXBarycenter, const rk *parentYBarycenter,
    const rk *parentField, const rk *parentCenter, const rk *parentXGrad, const rk *parentYGrad,
    const rk *grandField, const rk *grandCenter, const rk *grandXGrad, const rk *grandYGrad,
    int parentLayers, int grandLayers, int nParents, int hasChild,
    int nCells, int nCategories, int nLayers, int maxEdges)
{
    const int i = ir_thread();
    if (i >= nCells * nCategories * nLayers) return;
    const int layer = i % nLayers, cc = i / nLayers, c = cc / nCategories;
    const int stride = nCategories * nLayers;
    const int pi = cc * parentLayers + (parentLayers == 1 ? 0 : layer);
    const rk fc = field[i];
    // the parent's barycenter: the dummy tracer's geometric centre for the mass-like field
    const rk bx = nParents == 0 ? g0[c] : parentXBarycenter[pi];
    const rk by = nParents == 0 ? g1[c] : parentYBarycenter[pi];
    rk xg = C(0.0), yg = C(0.0);
    const int nEdges = nEdgesOnCell[c];
    if (maskCell[c] == 1) {
        // compute_gradient_2d/3d (4315-4405 / 4548-4642), planar
        for (int k = 0; k < nEdges; ++k) {
            const int n = cellsOnCell[c * maxEdges + k];
            rk normalGrad = C(0.0);
            if (n >= 1 && n <= nCells) {
                const int e = edgesOnCell[c * maxEdges + k] - 1;
                const int ni = (n - 1) * stride + (i - c * stride);
                if (mask[i] == 1 && mask[ni] == 1) {
                    const rk signGradient = (c + 1 == cellsOnEdge[2 * e]) ? C(1.0) : C(-1.0);
                    normalGrad = D(M(signGradient, S(field[ni], fc)), dcEdge[e]);
                }
            }
            xg = A(xg, M(coeffsReconstruct[(c * maxEdges + k) * 3], normalGrad));
            yg = A(yg, M(coeffsReconstruct[(c * maxEdges + k) * 3 + 1], normalGrad));
        }
        // limit_tracer_gradient_2d/3d (4905-5003 / 5117-5257)
        rk maxNeighbor = fc, minNeighbor = fc;
        for (int k = 0; k < nEdges; ++k) {
            const int n = cellsOnCell[c * maxEdges + k];
            const int ni = (n - 1) * stride + (i - c * stride);
            if (mask[ni] == 1) {
                maxNeighbor = ir_max(maxNeighbor, field[ni]);
                minNeighbor = ir_min(minNeighbor, field[ni]);
            }
        }
        maxNeighbor = S(maxNeighbor, fc);
        minNeighbor = S(minNeighbor, fc);
        rk maxLocal = C(0.0), minLocal = C(0.0);
        for (int k = 0; k < nEdges; ++k) {
            const rk deviationAtVertex = A(M(xg, S(xVertexOnCell[c * maxEdges + k], bx)),
                                           M(yg, S(yVertexOnCell[c * maxEdges + k], by)));
            maxLocal = ir_max(maxLocal, deviationAtVertex);
            minLocal = ir_min(minLocal, deviationAtVertex);
        }
        const rk gradFactor1 = ABS(maxLocal) > ABS(maxNeighbor) ? ir_max(C(0.0), D(maxNeighbor, maxLocal)) : C(1.0);
        const rk gradFactor2 = ABS(minLocal) > ABS(minNeighbor) ? ir_max(C(0.0), D(minNeighbor, minLocal)) : C(1.0);
        rk gradFactor = ir_min(gradFactor1, gradFactor2);
        gradFactor = ir_max(C(0.0), S(gradFactor, IR_EPS11));
        xg = M(xg, gradFactor);
        yg = M(yg, gradFactor);
    }
    xGrad[i] = xg;
    yGrad[i] = yg;
    // the value at the cell centre (3799-3805 / 3944-3983)
    const rk ce = S(S(fc, M(xg, bx)), M(yg, by));
    center[i] = ce;
    if (!hasChild || maskCell[c] != 1) return;
    const rk g[14] = {g0[c], g1[c], g2[c], g3[c], g4[c], g5[c], g6[c], g7[c], g8[c], g9[c], g10[c], g11[c],
                      g12[c], g13[c]};
    rk mean[3], cent[3], xs[3], ys[3];
    int n = 0;
    if (nParents == 2) {
        const int gi = cc * grandLayers + (grandLayers == 1 ? 0 : layer);
        mean[n] = grandField[gi]; cent[n] = grandCenter[gi]; xs[n] = grandXGrad[gi]; ys[n] = grandYGrad[gi]; ++n;
    }
    if (nParents >= 1) {
        mean[n] = parentField[pi]; cent[n] = parentCenter[pi]; xs[n] = parentXGrad[pi]; ys[n] = parentYGrad[pi];
        ++n;
    }
    mean[n] = fc; cent[n] = ce; xs[n] = xg; ys[n] = yg; ++n;
    ir_barycenter(n, mean, cent, xs, ys, g, &xBarycenter[i], &yBarycenter[i]);
}
