# Baseline and Logging

Baseline, MEAD, and hpctl all use logging extensively. hpctl uses it to collect scores from spawned jobs while baseline and mead use it to give the user information as well (more importantly) to report scores.

Default configuration can be found in [logging.json](../python/mead/config/logging.json).

## Loggers in Baseline and MEAD

 * `baseline.reporting`
   * This is what saves results of a given run.
   * This logs to a special `reporting-{pid}.log`
   * This logger should always log to the reporting file. It should not be disabled.
   * This logs at the `INFO` level
   * This outputs JSON messages
   * This also includes a console logger to log to stdout. This can be controlled separately from the file logging with the `REPORTING_LOG_LEVEL` environment variable.
 * `baseline.timing` This saves how long various parts of the system took.
   * This logs to a special `timing-{pid}.log`
   * This logs at the `DEBUG` level
   * This outputs JSON messages
   * This does not log to stdout
 * `baseline` This gives user information from baseline
   * This has a default config that logs json to console at `INFO` for use of baseline as an API.
   * This is overridden when used with a mead driver.
   * Mead driver default to having this write to `console`, `info.log` and `error.log`
   * Level can be controlled with `BASELINE_LOG_LEVEL` or `LOG_LEVEL`
   * These have a default config that is overridden by the mead logging
 * `mead` This gives user information from mead
   * This has a default config that logs json to console at `INFO` for use of baseline as an API.
   * This is overridden when used with a mead driver.
   * Mead driver default to having this write to `console`, `info.log` and `error.log`
   * Level can be controlled with `MEAD_LOG_LEVEL` or `LOG_LEVEL`


## Common options

 * Suppress logging from MEAD but not baseline `MEAD_LOG_LEVEL=WARNING`
 * Suppress logging of information from baseline but not mead `BASELINE_LOG_LEVEL=WARNING`
 * Suppress logging from MEAD and baseline but keep the reporting printouts. `MEAD_LOG_LEVEL=WARNING BASELINE_LOG_LEVEL=WARNING`
 * Suppress reporting from printing on screen but keep information from MEAD and baseline. `REPORTING_LOG_LEVEL=WARNING`
 * Suppress all logging `LOG_LEVEL=WARNING`
