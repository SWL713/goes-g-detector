Skip to content
SWL713
goes-g-detector
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
GOES G Detector
GOES G Detector #1
All jobs
Run details
Annotations
2 errors
detect-g
failed 1 minute ago in 25s
Search logs
1s
0s
0s
18s
3s
Run python src/run_g_detector.py
Traceback (most recent call last):
  File "/home/runner/work/goes-g-detector/goes-g-detector/src/run_g_detector.py", line 352, in <module>
    main()
  File "/home/runner/work/goes-g-detector/goes-g-detector/src/run_g_detector.py", line 324, in main
    chosen_trace, detections, chosen_source = pick_trace(resolved["east_df"], resolved["west_df"])
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/goes-g-detector/goes-g-detector/src/run_g_detector.py", line 259, in pick_trace
    east_candidates = detect_g_candidates(prep_trace(east_df), "GOES-East")
                                          ^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/goes-g-detector/goes-g-detector/src/run_g_detector.py", line 155, in prep_trace
    out = out.set_index("time_utc").resample("1min").median().interpolate(limit=5).reset_index()
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/resample.py", line 1478, in median
    return self._downsample("median", numeric_only=numeric_only)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/resample.py", line 2102, in _downsample
    result = obj.groupby(self._grouper).aggregate(how, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/groupby/generic.py", line 2291, in aggregate
    result = op.agg()
             ^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/apply.py", line 291, in agg
    return self.apply_str()
           ^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/apply.py", line 701, in apply_str
    return self._apply_str(obj, func, *self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/apply.py", line 792, in _apply_str
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/groupby/groupby.py", line 2385, in median
    result = self._cython_agg_general(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/groupby/groupby.py", line 1808, in _cython_agg_general
    new_mgr = data.grouped_reduce(array_func)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pandas/core/internals/managers.py", line 1646, in grouped_reduce
    applied = blk.apply(func)
              ^^^^^^^^^^^^^^^
0s
0s
0s
1s
0s
