"""Product admission for transport, low-level winds and the jet sequence."""
from woof.hex import products_door


def test_column_transport_and_jet_maps_are_default_products():
    # "ivt_column" is the renderer's model-column recipe; plain "ivt" is its 1000-300 hPa one, which a hex
    # history cannot feed.
    assert "ivt" not in products_door.GLOBAL_PRODUCTS
    assert {"ivt_column", "ivt_vectors", "isotherm_height_0c", "250mb_height_winds", "300mb_height_winds"} <= set(products_door.GLOBAL_PRODUCTS)
    assert {"925mb_height_winds", "850mb_height_winds"} <= set(products_door.REGIONAL_PRODUCTS)
    # the severe-weather set on regional views (rendered from the frames' own columns), not on global ones
    severe = {"mlcape", "srh_0_1km", "srh_0_3km", "bulk_shear_0_1km", "bulk_shear_0_6km", "ehi_0_1km", "stp_fixed",
              "lightning_threat_3", "10m_wind_gusts"}
    assert severe <= set(products_door.REGIONAL_PRODUCTS) and not severe & set(products_door.GLOBAL_PRODUCTS)
    assert len(products_door.GLOBAL_PRODUCTS) == len(set(products_door.GLOBAL_PRODUCTS))
    assert len(products_door.REGIONAL_PRODUCTS) == len(set(products_door.REGIONAL_PRODUCTS))
