from audit_fresh_scene_results import summarize


def test_incomplete_set_reports_subset_without_complete_score():
    rows=[dict(score_eligible=True,signed_residual_cm=-2),
          dict(score_eligible=False,signed_residual_cm=100)]
    result=summarize(rows,2)
    assert result['expected_regions']==2
    assert result['eligible_regions']==1
    assert result['complete_set_mae_cm'] is None
    assert result['eligible_subset_mae_cm']==2


def test_complete_set_uses_absolute_residuals_and_empty_is_unscored():
    assert summarize([dict(score_eligible=True,signed_residual_cm=-2),
                      dict(score_eligible=True,signed_residual_cm=4)],2)['complete_set_mae_cm']==3
    result=summarize([],2)
    assert result['complete_set_mae_cm'] is None
    assert result['eligible_subset_mae_cm'] is None
