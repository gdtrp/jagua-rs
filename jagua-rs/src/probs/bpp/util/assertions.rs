use crate::entities::Item;
use crate::probs::bpp::entities::{BPProblem, BPSolution, Bin};
use crate::util::assertions::snapshot_matches_layout;

pub fn problem_matches_solution(bpp: &BPProblem, sol: &BPSolution) -> bool {
    let BPSolution {
        layout_snapshots,
        time_stamp: _,
    } = sol;

    let bpp_density = bpp.density();
    let sol_density = sol.density(&bpp.instance);
    // Handle NaN case: both NaN is valid (e.g., when no items placed)
    assert!(
        (bpp_density.is_nan() && sol_density.is_nan()) || bpp_density == sol_density,
        "density mismatch: bpp={}, sol={}",
        bpp_density,
        sol_density
    );
    assert_eq!(bpp.layouts.len(), layout_snapshots.len());

    // Check that each layout in the problem has a matching snapshot in the solution
    bpp.layouts.iter().all(|(_, l)| {
        layout_snapshots
            .iter()
            .any(|(_, ls)| snapshot_matches_layout(l, ls))
    });

    true
}

pub fn instance_item_bin_ids_correct(items: &[(Item, usize)], bins: &[Bin]) -> bool {
    items.iter().enumerate().all(|(i, (item, _))| item.id == i)
        && bins.iter().enumerate().all(|(i, bin)| bin.id == i)
}
