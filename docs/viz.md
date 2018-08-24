## Visualization of Training Jobs

### Using visdom

Here are step-by-step instructions for seeing training progress with visdom:

#### Install visdom
```
pip install visdom
```

#### Launch visdom

```
dpressel@dpressel:~/dev/work/baseline/python/mead$ python -m visdom.server
```
Now, navigate your browser to: http://localhost:8097/


#### Run mead job with visdom argument

```
mead-train --config config/sst2.json --visdom 1
```

Losses will plot as the training progresses.  You can compare runs by giving a `visdom_name` argument to the trainer

```
mead-train --config config/sst2.json --visdom 1 --visdom_name <name>
```



### Using tensorboard

*TODO*