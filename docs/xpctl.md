## xpctl

In [mead](mead.md) we have separate **tasks** such as classify or tagger. Each task can have multiple **experiments**, each corresponding to a separate model or different hyperparameters of the same model. An experiment is uniquely identified by the id. The configuration for an experiment is uniquely identified by hash(_sha1_) of the config file. 

After an experiment is done, use `xpctl` to report the results to a database server. Then use it to analyze, compare and export your experimental results. It can be used in `repl` mode (inside a shell) and `command` mode. Most examples shown here uses xpctl in the `command` mode: (`xpctl command argument(s) option(s)` or `xpctl --host <your-host> --port <your-port> --dbtype <your-db> command argument(s) option(s)` )

### Contents
- [**Dependencies**](#dependencies)
- [**Installation**](#installation)
- [**REPL mode and commands**](#repl-mode-and-commands)
    - [**Commands**](#commands)
    - [**Updating the database**](#updating-the-database)
          - [**Importing**](#importing)
          - [**Exporting**](#exporting)
          - [**Summary**](#summary)
- [**Workflow for running an experiment**](#workflow-for-running-an-experiment)

### Dependencies

`xpctl` requires a database to be installed locally or an accessible server. We currently support:  [mongodb](https://docs.mongodb.com/) and [postgresql](https://www.postgresql.org/)), but the base classes can be extended to support other databases. 

### Installation

In the `python` dir, run `./install_dev.sh xpctl`. Before installation, you can create a file `xpctlcred.json` at your `HOME` directory if you do not want to pass the parameters every time you start the command. The file should look like this:
```
{
 "user": <username>,
 "passwd": <password>,
 "dbhost": <dbhost>,
 "dbport": <dbport>
}

```

_dbhost_ is typically `localhost` and _dbport_ is `y`. Here `<username>`, `<password>` and `<dbhost>` should be passed as string. If you are using a database other than _mongo_, put the line `"dbtype":"<databasetype>"`, eg. `"dbtype":"postgresql"`.

If you use [docker for baseline](docker.md), `xpctl` will be automatically installed.
 
### REPL Mode and Commands
 
**Starting**: use `--host`,`--port`,`--user`, `password` and `dbtype` to specify the host, port, username, password and dbtype for the database server. Else, you can pass a config file with the option `--config`. `xpctl` assumes that the config file is located at `~/xpctlcred.json` (in which case you can just use the command `xpctl` w/o specifying any option) but it can be saved anywhere you want.
```
(dl) home:home$ xpctl --host localhost
setting dbhost to localhost dbport to y
db connection successful
Starting xpctl...
xpctl > 

```

#### Commands

Details about any command can be found by `xpctl <command name> --help` on the terminal or `help <command-name>` inside the `xpctl` repl.
 
##### Set up and general info

- **vars**:   Prints the value of system variables. 
```
xpctl > vars
[DB] type: postgresql, host: localhost, port: 5432, user: x
xpctl > 
```

##### Analysis

- **results**: Provides a statistical summary of the results for a problem. A problem is defined by a (task, dataset) tuple. For each config used in the task, shows the average, min, max and std dev and number of experiments done using config.
 
  Usage: `results [OPTIONS] TASK DATASET`

  Optionally: 
  - `--metrics`: choose metric(s) to show. results ll be sorted on the first metric. 
  - `--sort`: output all metrics but sort on one. 
  - `--nconfig`: shows the statistical summaries for the last (sorted by time decreasing) _n_ experimental results per config. .
  - `--n`: shows _n_ results. 
  - `--event_type`: show results for train/dev/test datasets. defaults to _test_. 
  - `--output`: put the results in an output file.

```
xpctl > results tagger conll-iobes
db mongo connection successful with [host]: x.x.x, [port]: port_num
db mongo connection successful with [host]: x.x.x, [port]: port_num
                                               f1                                              acc                                        
                                         num_exps      mean       std       min       max num_exps      mean       std       min       max
sha1                                                                                                                                      
33aee90cc68beb1649bafc14015aa3827f159776      4.0  0.910722  0.001481  0.909091  0.912517      4.0  0.979112  0.000581  0.978292  0.979556
5d482cd9cd5d3b03d115d6da848131f38bc6e529      1.0  0.912186       NaN  0.912186  0.912186      1.0  0.979471       NaN  0.979471  0.979471
6edbdaa261620facab48c639385b513ae2feb868      1.0  0.911538       NaN  0.911538  0.911538      1.0  0.979771       NaN  0.979771  0.979771
7f85e5c982491bf401b5fa83614ba4b0f94fe373      3.0  0.905433  0.001328  0.904122  0.906777      3.0  0.978571  0.000119  0.978464  0.978699
89c8238a4154bb1cd652dfde7e687919a9317b68      3.0  0.910762  0.002732  0.907820  0.913220      3.0  0.979606  0.000527  0.978999  0.979942
d96a35d18d6d68242a5bda05ff14938ce5c81269      1.0  0.919784       NaN  0.919784  0.919784      1.0  0.981378       NaN  0.981378  0.981378
xpctl > 
```

```
xpctl > results classify SST2 --metric f1 --metric acc
db mongo connection successful with [host]: x.x.x, [port]: port_num
db mongo connection successful with [host]: x.x.x, [port]: port_num
                                              acc                                               f1                                        
                                         num_exps      mean       std       min       max num_exps      mean       std       min       max
sha1                                                                                                                                      
09facce8e04d7e65da2d6761dc5f397f39983a80      1.0  0.875343       NaN  0.875343  0.875343      1.0  0.876429       NaN  0.876429  0.876429
1a870728e04470a07643c5d9cff33329c004751f      1.0  0.881384       NaN  0.881384  0.881384      1.0  0.877133       NaN  0.877133  0.877133
67105e2108885c5ee08e211537fbda602f2ba254      1.0  0.866008       NaN  0.866008  0.866008      1.0  0.871849       NaN  0.871849  0.871849
72f2cce2ee4bcc755e527e03e05788442b658355      1.0  0.851099       NaN  0.851099  0.851099      1.0  0.853117       NaN  0.853117  0.853117
8ab6ab6ee8fdf14b111223e2edf48750c30c7e51      5.0  0.875892  0.004545  0.871499  0.883580      5.0  0.876028  0.006098  0.871570  0.886752
9ceabcb89c8bcb5500371c1898238d2973b12cdc      1.0  0.881933       NaN  0.881933  0.881933      1.0  0.883342       NaN  0.883342  0.883342
bdd99c99536c33fd3e7c312e0cc16c23e9e225f8      1.0  0.876991       NaN  0.876991  0.876991      1.0  0.871854       NaN  0.871854  0.871854
e1ddffe85b9e26906ba50da8d4a617acc8b4d162      1.0  0.877540       NaN  0.877540  0.877540      1.0  0.880301       NaN  0.880301  0.880301
ff37ce73365a152a9a652b60ef8036ce23bd608c      2.0  0.876167  0.005048  0.872597  0.879736      2.0  0.879907  0.004127  0.876988  0.882825
xpctl > 
```

```
xpctl > results classify SST2 --metric f1 --metric acc --sort f1
db mongo connection successful with [host]: x.x.x, [port]: port_num
db mongo connection successful with [host]: x.x.x, [port]: port_num
                                              acc                                               f1                                        
                                         num_exps      mean       std       min       max num_exps      mean       std       min       max
sha1                                                                                                                                      
9ceabcb89c8bcb5500371c1898238d2973b12cdc      1.0  0.881933       NaN  0.881933  0.881933      1.0  0.883342       NaN  0.883342  0.883342
e1ddffe85b9e26906ba50da8d4a617acc8b4d162      1.0  0.877540       NaN  0.877540  0.877540      1.0  0.880301       NaN  0.880301  0.880301
ff37ce73365a152a9a652b60ef8036ce23bd608c      2.0  0.876167  0.005048  0.872597  0.879736      2.0  0.879907  0.004127  0.876988  0.882825
1a870728e04470a07643c5d9cff33329c004751f      1.0  0.881384       NaN  0.881384  0.881384      1.0  0.877133       NaN  0.877133  0.877133
09facce8e04d7e65da2d6761dc5f397f39983a80      1.0  0.875343       NaN  0.875343  0.875343      1.0  0.876429       NaN  0.876429  0.876429
8ab6ab6ee8fdf14b111223e2edf48750c30c7e51      5.0  0.875892  0.004545  0.871499  0.883580      5.0  0.876028  0.006098  0.871570  0.886752
bdd99c99536c33fd3e7c312e0cc16c23e9e225f8      1.0  0.876991       NaN  0.876991  0.876991      1.0  0.871854       NaN  0.871854  0.871854
67105e2108885c5ee08e211537fbda602f2ba254      1.0  0.866008       NaN  0.866008  0.866008      1.0  0.871849       NaN  0.871849  0.871849
72f2cce2ee4bcc755e527e03e05788442b658355      1.0  0.851099       NaN  0.851099  0.851099      1.0  0.853117       NaN  0.853117  0.853117
xpctl > 
```

```
xpctl > results classify SST2 --metric f1 --metric acc --sort f1 --n 1
 db mongo connection successful with [host]: x.x.x, [port]: port_num
 db mongo connection successful with [host]: x.x.x, [port]: port_num
                                               acc                                         f1                                  
                                          num_exps      mean std       min       max num_exps      mean std       min       max
 sha1                                                                                                                          
 9ceabcb89c8bcb5500371c1898238d2973b12cdc      1.0  0.881933 NaN  0.881933  0.881933      1.0  0.883342 NaN  0.883342  0.883342
 ff37ce73365a152a9a652b60ef8036ce23bd608c      1.0  0.879736 NaN  0.879736  0.879736      1.0  0.882825 NaN  0.882825  0.882825
 e1ddffe85b9e26906ba50da8d4a617acc8b4d162      1.0  0.877540 NaN  0.877540  0.877540      1.0  0.880301 NaN  0.880301  0.880301
 1a870728e04470a07643c5d9cff33329c004751f      1.0  0.881384 NaN  0.881384  0.881384      1.0  0.877133 NaN  0.877133  0.877133
 09facce8e04d7e65da2d6761dc5f397f39983a80      1.0  0.875343 NaN  0.875343  0.875343      1.0  0.876429 NaN  0.876429  0.876429
 bdd99c99536c33fd3e7c312e0cc16c23e9e225f8      1.0  0.876991 NaN  0.876991  0.876991      1.0  0.871854 NaN  0.871854  0.871854
 67105e2108885c5ee08e211537fbda602f2ba254      1.0  0.866008 NaN  0.866008  0.866008      1.0  0.871849 NaN  0.871849  0.871849
 8ab6ab6ee8fdf14b111223e2edf48750c30c7e51      1.0  0.871499 NaN  0.871499  0.871499      1.0  0.871570 NaN  0.871570  0.871570
 72f2cce2ee4bcc755e527e03e05788442b658355      1.0  0.851099 NaN  0.851099  0.851099      1.0  0.853117 NaN  0.853117  0.853117
 xpctl > 

```

```
xpctl > results classify SST2 --metric f1 --metric acc --sort f1 --n 1
db mongo connection successful with [host]: x.x.x, [port]: port_num
db mongo connection successful with [host]: x.x.x, [port]: port_num
                                               f1                                        acc                                  
                                         num_exps      mean std       min       max num_exps      mean std       min       max
sha1                                                                                                                          
9ceabcb89c8bcb5500371c1898238d2973b12cdc      1.0  0.883342 NaN  0.883342  0.883342      1.0  0.881933 NaN  0.881933  0.881933
xpctl > 

```

- **details**: Shows the results for all experiments for a particular config (sha1). Optionally filter out by user(s), metric(s), or sort by one metric. Shows the results on the test data by default, provide event_type (train/valid/test) to see for other datasets. Optimally limit the number of results shown. 

  Usage: `details [OPTIONS] TASK SHA1`
  
  Optionally: 
  - `--metrics`: choose metric(s) to filter the results on. results ll be sorted on the first metric. 
  - `--sort`: output all metrics but sort on one. 
  - `--n`: shows the last (by time) _n_ experimental results. 
  - `--event_type`: show results for train/dev/test datasets. defaults to _test_. 

```
xpctl > results tagger conll-iobes
db mongo connection successful with [host]: x.x.x, [port]: num_port
db mongo connection successful with [host]: x.x.x, [port]: num_port
                                              acc                                               f1                                        
                                         num_exps      mean       std       min       max num_exps      mean       std       min       max
sha1                                                                                                                                      
33aee90cc68beb1649bafc14015aa3827f159776      4.0  0.979112  0.000581  0.978292  0.979556      4.0  0.910722  0.001481  0.909091  0.912517
5d482cd9cd5d3b03d115d6da848131f38bc6e529      1.0  0.979471       NaN  0.979471  0.979471      1.0  0.912186       NaN  0.912186  0.912186
6edbdaa261620facab48c639385b513ae2feb868      1.0  0.979771       NaN  0.979771  0.979771      1.0  0.911538       NaN  0.911538  0.911538
7f85e5c982491bf401b5fa83614ba4b0f94fe373      3.0  0.978571  0.000119  0.978464  0.978699      3.0  0.905433  0.001328  0.904122  0.906777
89c8238a4154bb1cd652dfde7e687919a9317b68      3.0  0.979606  0.000527  0.978999  0.979942      3.0  0.910762  0.002732  0.907820  0.913220
d96a35d18d6d68242a5bda05ff14938ce5c81269      1.0  0.981378       NaN  0.981378  0.981378      1.0  0.919784       NaN  0.919784  0.919784
xpctl > details tagger 33aee90cc68beb1649bafc14015aa3827f159776
db mongo connection successful with [host]: x.x.x, [port]: num_port
db mongo connection successful with [host]: x.x.x, [port]: num_port
                          id  username           label      dataset                                      sha1                       date       acc        f1
44  5b2cdc2af5ed250de2b5dc45  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 11:23:22.497571  0.978292  0.910056
45  5b2cdc35f5ed250e6bd54c13  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 11:23:33.690791  0.979492  0.912517
46  5b2d0159f5ed2557e3db8859  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 14:02:01.587844  0.979106  0.909091
47  5b2d0169f5ed255b01e6da8c  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 14:02:17.444505  0.979556  0.911225
xpctl > details tagger 33aee90cc68beb1649bafc14015aa3827f159776 --sort f1
db mongo connection successful with [host]: x.x.x, [port]: num_port
db mongo connection successful with [host]: x.x.x, [port]: num_port
                          id  username           label      dataset                                      sha1                       date       acc        f1
45  5b2cdc35f5ed250e6bd54c13  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 11:23:33.690791  0.979492  0.912517
47  5b2d0169f5ed255b01e6da8c  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 14:02:17.444505  0.979556  0.911225
44  5b2cdc2af5ed250de2b5dc45  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 11:23:22.497571  0.978292  0.910056
46  5b2d0159f5ed2557e3db8859  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 14:02:01.587844  0.979106  0.909091
xpctl > details tagger 33aee90cc68beb1649bafc14015aa3827f159776 --sort f1 --n 2
db mongo connection successful with [host]: x.x.x, [port]: num_port
db mongo connection successful with [host]: x.x.x, [port]: num_port
                          id  username           label      dataset                                      sha1                       date       acc        f1
47  5b2d0169f5ed255b01e6da8c  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 14:02:17.444505  0.979556  0.911225
46  5b2d0159f5ed2557e3db8859  dpressel  conll-iobes-tf  conll-iobes  33aee90cc68beb1649bafc14015aa3827f159776 2018-06-22 14:02:01.587844  0.979106  0.909091
xpctl > 
```
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
setting dbhost to xxx.yyy.com dbport to y
db connection successful
Usage: xpctl delete [OPTIONS] TASK ID
  delete a record from database with the given object id. also delete the
  associated model file from the checkpoint.
Options:
  --help  Show this message and exit.
```

##### Importing

- **putresult**: puts the result of an experiment in the database. Can optionally store the model files in a persistent model store (will automatically zip them). 
```
xpctl > help putresult
Usage: putresult [OPTIONS] TASK CONFIG LOG LABEL
  Puts the results of an experiment on the database. Arguments:  task name (classify/ tagger), location of the config file for the experiment, the log file storing the
  results for the experiment (typically <taskname>/reporting.log) and a
  short description of the experiment (label). Gets the username from system
  (can provide as an option). Also provide the model location produced by
  the config optionally. 
Options:
  --user TEXT    username
  --cbase TEXT   path to the base structure for the model checkpoint
                 files:such as ../tagger/tagger-model-tf-11967 or
                 /home/ds/tagger/tagger-model-tf-11967 
  --cstore TEXT  location of the model checkpoint store (default <store-path> in your machine)
  --help         Show this message and exit.
```

```
xpctl > putresult --user ijindal classify <path>/config/sst2.json <path>/reporting-4923.log  testClassify

db mongo connection successful with [host]: x.x.x, [port]: y
updating results for existing task [classify] in host [x.x.x]
results updated, the new results are stored with the record id: 5b1aacb933ed5901dc545af8

```
This record id is then used in *putmodel*

- **putmodel**: save model files in a persistent location. The location can be provided by the option -cstore, by default it is `<store-path>` directory in your machine. This is tested for `tensorflow` models, not `pytorch` ones yet. 
```
xpctl > putmodel --help
Usage: putmodel [OPTIONS] TASK ID CBASE
  Puts the model from an experiment in the model store and updates the
  database with the location. Arguments:  task name (classify/ tagger), record id, and
  the path to the base structure for the model checkpoint files such as
  ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967
Options:
  --cstore TEXT  location of the model checkpoint store
  --help         Show this message and exit.

```

```
 xpctl > putmodel classify 5b1aacb933ed5901dc545af8 <path>/classify-model-tf-4923/classify-model-tf-4923

 db mongo connection successful with [host]: x.x.x, [port]: y
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.saver] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.model.index] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.graph] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.model.data-00000-of-00001] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.model.meta] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.labels] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
writing model file: [<path>/classify-model-tf-4923/classify-model-tf-4923.vocab] to store: [<store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1]
zipping model files
zipped file written, model directory removed
database updated with <store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1.zip

```
##### Exporting

- **getmodelloc** : shows the model location for an id (**id, not SHA1**. An experiment can be run multiple times using the same config). 
```
xpctl > help getmodelloc
Usage: xpctl getmodelloc [OPTIONS] TASK ID
  get the model location for a particular task name (classify/ tagger) and record id
Options:
  --help  Show this message and exit.
```

```
xpctl > getmodelloc classify 5b1aacb933ed5901dc545af8

db mongo connection successful with [host]: x.x.x, [port]: y
db mongo connection successful with [host]: x.x.x, [port]: y
model loc is <store-path>/67105e2108885c5ee08e211537fbda602f2ba254/1.zip

```

- **config2json**
```
xpctl > help config2json
Usage: config2json [OPTIONS] TASK SHA FILENAME
  Exports the config file for an experiment as a json file. Arguments:
  task name (classify/ tagger), sha1 for the experimental config, output file path
Options:
  --help  Show this message and exit.
```
Here `sha1` is the model-checkpoint id.

```
xpctl > config2json classify 67105e2108885c5ee08e211537fbda602f2ba254 <path>/c_SST2.json

db mongo connection successful with [host]: x.x.x, [port]: y
db mongo connection successful with [host]: x.x.x, [port]: y

```

##### Summary


- **lbsummary**: provides a description of all tasks in the leaderboard. 
```
xpctl > lbsummary --help
Usage: lbsummary [OPTIONS]
  Provides a summary of the leaderboard. Options: taskname. If you provide a
  taskname, it will show all users, and datasets for that task. 
Options:
  --task TEXT
  --help       Show this message and exit.

```

```
xpctl > lbsummary --task tagger
Task: [tagger]
---------------------------------------------------------------------------------------------
                                 task
                             num_exps
user         dataset                 
blester      conll                  1
digitalroots wnut                   2
dpressel     conll                  3
             conll-bio              5
             conll-iobes           13
             sf-ivt                 1
             twpos                  1
             wnut                   5
mbarta       sf-ivt-20180607        3
root         conll                  1
             idr-05-14              1
             idr-18-07-11          10
             pj-intents-v2.0        1
             sf-idr-pj-08-16        6
             slftemplate            1
             wnut                  12
schoudhury   idr                    7
             idr-05-14              1
             idr-18-07-11           1
             idr-fca                2
             sf-idr-pj-08-09        1
             sf-pj                  1
             wnut                   5
xpctl > 
```


### Workflow for Running an Experiment

Perform an experiment E, i.e., train the model and test. 

Using xpctl 

1. Put the results ( `putresult`). This will show you the id of the result you just updated in the database. 
2. Get the best average result by a config so far: `xpctl results <task> <test-dataset> --sort <metric> --n 1`. This will give you the required sha1 for the best config.
3. Get details for that config: `xpctl details <task> <sha1> --sort f1`. This will give you the id for the best performing experiment. Let's call it E'.
4. If E' == E, your current experiment is the best so far, use `putmodel` to store the model in a persistent loc.

_ sometime later _ : 

5. Check the best average result by a config so far: `xpctl results tagger <test-dataset> --sort f1 --n 1`
6. Get details for that config: `xpctl details tagger <sha1> --sort f1`. This will give you the id for the best performing model.  
7. To check if we have the model stored in a persistent loc: `getmodelloc <id>`.
9. Use the model files in your system. 
