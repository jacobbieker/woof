"""QNSE and BouLac local frames are priced at the tier the launcher compiles."""

from woof.core import preflight


def test_qnse_and_boulac_price_both_tiers():
    for module, shipped in (("qnsepbl", 16896), ("boulac", 8208)):
        frame, select_tier = preflight.CONDITIONAL_COLUMN_TIERED_FRAMES[module]
        assert preflight.KERNEL_MAX_LOCAL_SIZE_BYTES[module] == shipped
        assert frame.frame_bytes(select_tier(49)) == shipped
        assert frame.frame_bytes(select_tier(128)) == shipped
        assert frame.frame_bytes(select_tier(129)) == frame.frame_bytes(256)
        assert frame.frame_bytes(select_tier(256)) > shipped
