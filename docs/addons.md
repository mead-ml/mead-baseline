## Extending Baseline and MEAD

Baseline provides a simple extension process made up of registering handlers for some aspect of training using a python decorator.  There are hooks for overriding many of the functionalities inside of Baseline without a lot of hassle.

These extensions can be used programmatically if you are not using mead at all.  In that case, use mead.tasks as a reference for how you can orchestrate the code

Almost all extensions are done by writing a class to handle some aspect of training, and decorating it with a `@register_XX` hook, where `XX` is the aspect of training to override.  The currently supported extension points are:

- `@register_model(cls, task, name=None)`: The task will be one of the supported baseline tasks (e.g. `'classify'`).  The name will be a user-defined key, or the class name if none is given, which will be used to identify it in the mead config (or calling program)

- `@register_vectorizer(cls, name=None)`: Create your own vectorizer, and give it a key (name) which can be used to identify it in the mead config.  Implementors should inherit from `baseline.Vectorizer`

- `@register_embeddings(cls, name=None)`: Create your own embeddings sub-graph in the DL framework you are using.  The name is a key which is use to identify the embeddings in mead.  As in the other cases, it defaults to the class name.  Implementors should inherit from `baseline.{framework}.Embeddings`

- `@register_reporting(cls, name=None)`: Create your own reporting hook which will be advised of training updates. The name, if none, deafults to the class name.  There is a parameter block that will be passed to this constructor to initialize it by mead.  Implementors should inherit from `baseline.ReportingHook`

- `@register_reader(cls, name=None)`: Create your own readers for given tasks.  The name is defaulted to the reader class name

- `@register_trainer(cls, name=None)`: Create your own trainer which will be typically be employed by the `fit()` function.  Note that its possible to override the actual function used for `fit()`, but this is not recommended in most cases.  Implementors should typically inherit from `baseline.{framework}.Trainer` 

- `@register_training_func(cls, task, name)`: This is advanced functionality which bypasses the usual `baseline.{framework}.{task}.fit()` function allowing the user to completely define their own training mechanism which is a block box to Baseline/MEAD

### Inversion of Control and MEAD

MEAD is a program that orchestrates training by delegation of its sub-components to registered handlers.  This is a design pattern known as Inversion of Control (IoC), and its commonly used in plugin architectures, and very powerful, but it can be confusing to the uninitiated.  The basic idea in our case is this:  we know we want to use some generalized training routine to use some DL framework to train some sort of model for some task.  We want to also properly vectorize inputs and readers.  The typical pattern for this would be to create some pre-registered components for everything, but to allow a user to define their own implementations, and somehow know that these exists and should be called to handle certain events.

The registration process in Baseline makes this easy for MEAD.  All the classes which a researcher wants to use for some hook that MEAD needs to call have to register themselves with a dictionary (usually referred to as a registry).  But wwe want to make this as simple as possible for users.

Previous versions required the user to have a python module somewhere with a convention-based name and some special `create_` and `load_` methods to be exposed to MEAD.  This is nice because it encapsulates underlying module implementation from MEAD, but its clunky because it requires a user to have to think about more than just a class for some job.  The new registration process is much simpler, requiring a single line of code to decorate the class.  These classes can still be used externally, but when used in `mead-train` with it inverted control, MEAD can still find all the hooks the user specifies in the mead config and execute them.

This also simplifies the codebase significantly removing lots of boilerplate import and resolution code from Baseline.

Only one issue remains -- how to know where to find user code.  Obviously we should be able to assume that code lives in the `PYTHONPATH`, but how should we know there is something the user wishes to load without modifying `mead-train`?

The answer is actually quite simple -- we tell MEAD what additional modules to import from the mead config.  Unlike previous versions where you needed some canonical file named something like `classify_rnf.py` (indicating this is a `classify` task handler named `rnf`), lets imagine we have a whole user-defined library of things living in a file called `userlib.py`.  We can simply tell mead, load `userlib.py`, and as long as we have a classifier registered to `rnf`, we can tell mead to use `model_type: rnf` as we did previously.

### A Simple Example

- [Here is a model](../python/addons/rnf_pyt.py) we want to train.  Its based off: https://github.com/bloomberg/cnn-rnf but I have rewritten it in PyTorch as a simple single class inheriting from `ClassifierModelBase` (just to remove some boilerplate code).

Its defined as a simple class with a decorator that will register the class with Baseline.  The decorator tells Baseline that this class is going to handle any training instantion where the `model_type` given is 'rnf'.

- [Here is the config](../python/mead/config/sst2-rnf-pyt.yml) that can be used in MEAD to run train this model
  - There are two points of interest
    1. the `model_type` is defined as 'rnf'
    2. the `modules` section (a list), tells us python modules to load into MEAD

- Now all I have to do is call `mead-train` (trainer.py)

```
python trainer.py --config config/sst2-rnf-pyt.yml
```

We can put our code in any python module that we wish -- for instance, we might have library of registered hooks for training, models and readers.  We can just tell mead about the library in the `modules` section, and then through proper decoration, they become available for training.  Here is an example of a module that customizes several aspects of training:
  - https://github.com/dpressel/baseline/blob/feature/v1/python/addons/demolib.py
And here is the corresponding YAML to run this in mead:

  - https://github.com/dpressel/baseline/blob/feature/v1/python/mead/config/sst2-demolib.yml
