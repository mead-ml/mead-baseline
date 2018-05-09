## xpctl

In [mead](mead.md) we have separate **tasks** such as classify or tagger. Each task can have multiple **experiments**, each corresponding to a separate model or different hyperparameters of the same model. An experiment is uniquely identified by the id. The configuration for an experiment is uniquely identified by hash(_sha1_) of the config file. 

After an experiment is done, use `xpctl` to report the results to a mongodb server. Then use it to analyze, compare and export your experimental results. It can be used in `repl` mode (inside a shell) and `command` mode. Most examples shown here uses xpctl in the `command` mode: (`xpctl command arguments options` or `xpctl --host localhost --port 27107 command argument(s) option(s)` )

### Contents
- [**Dependencies**](#dependencies)
- [**Installation**](#installation)
- [**REPL mode and commands**](#repl-mode-and-commands)
- [**Workflow for running an experiment**](#workflow-for-running-an-experiment)

### Dependencies

`xpctl` requires [mongodb](https://docs.mongodb.com/) to be installed locally or an accessible server.

### Installation

In the `xpctl` dir, run `pip install -e`. Before installation, you can create a file `xpctlcred.json` at your `HOME` directory if you do not want to pass the parameters everytime you start the command. The file should look like this:

```
{
 "user": <username>,
 "passwd": <password>,
 "dbhost": <dbhost>,
 "dbport": <dbport>
}

```

_dbhost_ is typically `localhost` and _dbport_ is `27017`. 

If you use [docker](docker.md), `xpctl` will be automatically installed.
 
### REPL Mode and Commands
 
**Starting**: use `--host`,`--port`,`--user` and `password` to specify the host, port, username and password for mongodb. Else, you can pass a config file with the option `--config`. `xpctl` assumes that the config file is located at `~/xpctlcred.json` (in which case you can just use the command `xpctl` w/o specifying any option) but it can be saved anywhere you want.

 ```
(dl) home:home$ xpctl --host localhost
setting dbhost to localhost dbport to 27017
db connection successful
Starting xpctl...
xpctl > 
```
#### Commands

##### Set up and general info

- **vars**:   Prints the value of system variables dbhost and dbport. 
```
xpctl > vars
dbhost: xxx.yyy.com, dbport: 27017
xpctl >
```

##### Analysis

- **results**: shows results for a task, event type (train/test/valid) and a dataset. Optionally supply metric(s) and a metric to sort the results with. If you specify **only one** metric, the results will be sorted on that.

```
Usage: xpctl results [OPTIONS] TASK EVENT_TYPE DATASET
  Shows the results for event_type(tran/valid/test) on a particular task
  (classify/ tagger) with a particular dataset (SST2, wnut). Default
  behavior: **All** users, and metrics. Optionally supply user(s),
  metric(s), a sort metric. use --metric f1 --metric acc for multiple
  metrics. use one metric for the sort.
Options:
  --user TEXT    list of users (test-user, root), [multiple]: --user a --user b
  --metric TEXT  list of metrics (prec, recall, f1, accuracy),[multiple]:
                 --metric f1 --metric acc
  --sort TEXT    specify one metric to sort the results
  --help         Show this message and exit.
```

```
xpctl > results classify test SST2
                         id  username                    label dataset                                      sha1                        date  avg_loss        f1    recall       acc  precision
0  5af36f9bb5536c60d1e2ccc1  dpressel  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T22:00:49.195327  0.306686  0.873474  0.865787  0.874794   0.881299
1  5af34c0bb5536c533c9b6ecc      None  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T19:29:15.696297  0.293442  0.871570  0.873487  0.871499   0.869660
2  5af34c9fb5536c53bb07bc46      None  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T19:31:42.984408  0.321655  0.874655  0.871287  0.875343   0.878049
```

```
xpctl > results classify test SST2 --metric acc
                         id  username                    label dataset                                      sha1                        date       acc
2  5af34c9fb5536c53bb07bc46      None  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T19:31:42.984408  0.875343
0  5af36f9bb5536c60d1e2ccc1  dpressel  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T22:00:49.195327  0.874794
1  5af34c0bb5536c533c9b6ecc      None  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T19:29:15.696297  0.871499
```

```
(xpt2.7) testuser:~$ xpctl results tagger test wnut --metric f1 --metric acc --sort acc
db connection successful with [host]: x.y.com, [port]: 27017
                          id      username                                      label dataset                                      sha1                    date       acc        f1
4   5a1611c038d2dd7832f7c47c    testuser                      bilstm-CRF-filt-1-2-3    wnut  39831513cb319984c9410a2c7da9429ec0bc745d 2017-11-23 00:09:33.455  0.939854  0.391667
8   5a02665d6232a6030a275fc3          root     wnut-lowercased-gazetteer-5proj-unif-0    wnut  4073e6dff65340d954af3e66d36a6d144fca5825 2017-11-08 02:05:17.009  0.939287  0.386049
```

```
(xpt2.7) testuser:~$ xpctl results tagger test wnut --metric f1 --metric acc --sort f1
db connection successful with [host]: x.y.com, [port]: 27017
                          id      username                                      label dataset                                      sha1                    date       acc        f1
10  5a05f17c6232a638b53903b0  testuser             wnut-gazetteer-noproj-unif-0.1    wnut  ab7b1fed1f0206ad9ebf3887657f18051d99f643 2017-11-10 18:35:40.741  0.936957  0.402537
6   5a0140636232a60282e0cb8d          root  wnut-lowercased-gazetteer-noproj-unif-0.1    wnut  231c11c31fa8389a192da9e84a1c19d7ada46613 2017-11-07 05:10:59.629  0.938211  0.395871
9   5a026c6a6232a605d4ea2688          root    wnut-lowercased-gazetteer-noproj-unif-0    wnut  ab7b1fed1f0206ad9ebf3887657f18051d99f643 2017-11-08 02:31:06.897  0.938391  0.391900
```

```
(xpt2.7) testuser:~$ xpctl results tagger test wnut --metric f1
db connection successful with [host]: x.y.com, [port]: 27017
                          id      username                                      label dataset                                      sha1                    date        f1
10  5a05f17c6232a638b53903b0  testuser             wnut-gazetteer-noproj-unif-0.1    wnut  ab7b1fed1f0206ad9ebf3887657f18051d99f643 2017-11-10 18:35:40.741  0.402537
6   5a0140636232a60282e0cb8d          root  wnut-lowercased-gazetteer-noproj-unif-0.1    wnut  231c11c31fa8389a192da9e84a1c19d7ada46613 2017-11-07 05:10:59.629  0.395871
9   5a026c6a6232a605d4ea2688          root    wnut-lowercased-gazetteer-noproj-unif-0    wnut  ab7b1fed1f0206ad9ebf3887657f18051d99
```

- **best**: shows the best (n) result for a task. 

```
xpctl > help best
Usage: best [OPTIONS] TASK EVENT_TYPE DATASET METRIC
  Shows the best F1 score for event_type(tran/valid/test) on a particular
  task (classify/ tagger) on a particular dataset (SST2, wnut) using a
  particular metric. Default behavior: The best result for **All** users
  available for the task. Optionally supply number of results (n-best),
  user(s) and metric(only ONE)
Options:
  --user TEXT  list of users (test-user, root), [multiple]: --user a --user b
  --n INTEGER  N best results
  --help       Show this message and exit.
```

```
xpctl > best --n 3 classify test SST2 acc

total 3 results found, showing best 3 results
                         id  username                    label dataset                                      sha1                        date       acc
2  5af34c9fb5536c53bb07bc46      None  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T19:31:42.984408  0.875343
0  5af36f9bb5536c60d1e2ccc1  dpressel  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6ee8fdf14b111223e2edf48750c30c7e51  2018-05-09T22:00:49.195327  0.874794
1  5af34c0bb5536c533c9b6ecc      None  Kim2014-SST2-TF-2epochs    SST2  8ab6ab6e```

#### Updating the database

- **updatelabel**: update the label for an experiment.
```
xpctl > help updatelabel
Usage: updatelabel [OPTIONS] TASK ID LABEL
  Update the label for an experiment (identified by its id) for a task
Options:
  --help  Show this message and exit.
```

- **delete**: deletes a record from the database and the associated model file from model-checkpoints if it exists.
```
setting dbhost to xxx.yyy.com dbport to 27017
db connection successful
Usage: xpctl delete [OPTIONS] TASK ID
  delete a record from database with the given object id. also delete the
  associated model file from the checkpoint.
Options:
  --help  Show this message and exit.
```

- **putresult**: puts the result of an experiment in the database. Can optionally store the model files in a persistent model store (will automatically zip them) This is tested for `tensorflow` models, not `pytorch` ones yet. 
 
```
xpctl > help putresult
Usage: putresult [OPTIONS] TASK CONFIG LOG LABEL
  Puts the results of an experiment on the database. Arguments:  task name,
  location of the config file for the experiment, the log file storing the
  results for the experiment (typically <taskname>/reporting.log) and a
  short description of the experiment (label). Gets the username from system
  (can provide as an option). Also provide the model location produced by
  the config optionally. 
Options:
  --user TEXT    username
  --cbase TEXT   path to the base structure for the model checkpoint
                 files:such as ../tagger/tagger-model-tf-11967 or
                 /home/ds/tagger/tagger-model-tf-11967 
  --cstore TEXT  location of the model checkpoint store (default /data/model-checkpoints in your machine)
  --help         Show this message and exit.
```

- **putmodel**: save model files in a persistent location. The location can be provided by the option -cstore, by default it is `/data/model-checkpoints` directory in your machine. This is tested for `tensorflow` models, not `pytorch` ones yet. 

```
xpctl > putmodel --help
Usage: putmodel [OPTIONS] TASK ID CBASE
  Puts the model from an experiment in the model store and updates the
  database with the location. Arguments:  task name, id of the record, and
  the path to the base structure for the model checkpoint files such as
  ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967
Options:
  --cstore TEXT  location of the model checkpoint store
  --help         Show this message and exit.

```

##### Exporting

- **getmodelloc** : shows the model location for an id (**id, not SHA1**. An experiment can be run multiple times using the same config). 
```
xpctl > help modelloc
Usage: xpctl getmodelloc [OPTIONS] TASK ID
  get the model location for a particluar task and record id
Options:
  --help  Show this message and exit.
```

- **config2json**
```
xpctl > help config2json
Usage: config2json [OPTIONS] TASK SHA FILENAME
  Exports the config file for an experiment as a json file. Arguments:
  taskname, sha1 for the experimental config, output file path
Options:
  --help  Show this message and exit.
```

##### Summary

- **tasksummary**: provides a natural language summary for a task. 

```
xpctl > help tasksummary
Usage: tasksummary [OPTIONS] TASK DATASET METRIC
  Provides a natural language summary for a task. This is almost equivalent
  to the `best` command.
Options:
  --help  Show this message and exit.
```

```
xpctl > tasksummary classify sa180k macro_f1
For dataset sa180k, the best f1 is 0.718 reported by root on 2017-09-08 17:24:32.963000. The sha1 for the config file is 6dac0ea88618ab67d6f5d690279c13f1ec305167.
```

- **lbsummary**: provides a description of all tasks in the leaderboard. 

```
xpctl > lbsummary --help
Usage: lbsummary [OPTIONS]
  Provides a summary of the leaderboard. Options: taskname. If you provide a
  taskname, it will show all users, datasets, event_types and metrics for that task. This
  is helpful because we often forget what metrics or datasets were used for
  a task, which are the necessary parameters for the commands `results` and
  `best` and `tasksummary`. Shows the summary for all available tasks if no
  option is specified.
Options:
  --task TEXT
  --help       Show this message and exit.

```
```
xpctl > lbsummary --task tagger
Task: [tagger]
---------------------------------------------------------------------------------------------
         user dataset    event_type   metrics  num_experiments
0    test-user   twpos   test_events    acc,f1                1
1    test-user   twpos  train_events  avg_loss                1
2    test-user   twpos  valid_events    acc,f1                1
6  testuser    wnut   test_events    acc,f1                1
7  testuser    wnut  train_events  avg_loss                1
8  testuser    wnut  valid_events    acc,f1                1
```


### Workflow for Running an Experiment

Perform an experiment E, i.e., train the model and test. 

Using xpctl 
1. Put the results ( `putresult`). This will show you the id of the result you just updated in the database. 
2. Get the best result so far: `xpctl best tagger test <test-dataset> f1`
3. If E is the best so far, use `putmodel` to store the model in a persistent loc.

_ sometime later _ : 
4. Check the best model so far: `best`
4. To check if we have the model stored in a persistent loc: `getmodelloc`.

```
(dl) testuser:xpctl$ xpctl best tagger test <test-dataset> f1
setting dbhost to localhost dbport to 27017
db connection successful
using metric: ['f1']
total 2 results found, showing best 1 results
                         id    username label dataset                                      sha1                    date        f1
1  59b9af747412df155562438d  testuser  test     <test-dataset>  7fbbd90b57395b003ab8476b6a17e747f8dfcba3 2017-09-13 22:21:35.322  0.860052
(dl) testuser:xpctl$ xpctl putmodel --help
setting dbhost to localhost dbport to 27017
db connection successful
Usage: xpctl putmodel [OPTIONS] TASK ID CBASE

  Puts the model from an experiment in the model store and updates the
  database with the location. Arguments:  task name, id of the record, and
  the path to the base structure for the model checkpoint files such as
  ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967

Options:
  --cstore TEXT  location of the model checkpoint store
  --help         Show this message and exit.
(dl) testuser:xpctl$ xpctl getmodelloc --help
setting dbhost to localhost dbport to 27017
db connection successful
Usage: xpctl getmodelloc [OPTIONS] TASK ID

  get the model location for a particluar task and record id

Options:
  --help  Show this message and exit.

(dl) testuser:xpctl$ xpctl getmodelloc tagger 59b9af747412df155562438d
setting dbhost to localhost dbport to 27017
db connection successful
['model storage loc for record id 59b9af747412df155562438d is /data/model-checkpoints/7fbbd90b57395b003ab8476b6a17e747f8dfcba3/1.zip']

```
