"""icuflow — the parts of a clinical prediction pipeline that are easy to get wrong.

Three of them, packaged because they are re-implemented badly in most clinical
prediction code:

  splits        grouping by patient, so a subject never spans train and test
  calibration   turning a score into a probability that means what it says
  alarms        comparing alerting systems at matched sensitivity

Designed to sit alongside PyHealth and RHealth rather than replace them: the
functions take and return plain arrays, so they drop into an existing task
pipeline without adopting a framework.
"""

from .splits import (GroupSplit, group_kfold, split_by_subject,
                     assert_no_subject_leak)
from .calibration import (IsotonicCalibrator, PlattCalibrator, ece,
                          reliability_table)
from .alarms import (alarm_stats, threshold_at_sensitivity,
                     compare_at_matched_sensitivity)
from .tasks import horizon_label, sliding_windows

__version__ = "0.1.0"

__all__ = [
    "GroupSplit", "group_kfold", "split_by_subject", "assert_no_subject_leak",
    "IsotonicCalibrator", "PlattCalibrator", "ece", "reliability_table",
    "alarm_stats", "threshold_at_sensitivity", "compare_at_matched_sensitivity",
    "horizon_label", "sliding_windows", "__version__",
]
