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

### Addons and mead-hub

Addons that live on mead-hub (or any other URL) can be downloaded and used for training automatically.  They are downloaded
into the user's data-cache, which defaults to `~/.bl-data` and an entry will be recorded in the data-cache index which
is located inside the data-cache in a file called `data-cache.json`.  The addons will be automatically added to the user's
import path for training.  For example, the example [demolib module](https://github.com/mead-ml/hub/blob/master/v1/addons/demolib.py)
can be referenced from the mead config as `hub:v1:addons:demolib` as in [this example](../mead/config/sst2-demolib.yml).

The library will download this path into the cache, usually at `~/.bl-data/addons/demolib.py` and it will add it to a field called `hub_modules`
in the model (for TensorFlow, in PyTorch the entire module is persisted already). 

When we go to reload the model for inference, the module will automaticall be re-added to the data-cache if its not present ni the model.

