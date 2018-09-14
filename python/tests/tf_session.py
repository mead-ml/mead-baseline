# borrowed from GPFlow
import functools
from contextdecorator import ContextDecorator
import tensorflow as tf
import pytest
import os

@pytest.fixture
def session_tf():
    """
    Session creation pytest fixture.
    ```
    def test_simple(session_tf):
        tensor = tf.constant(1.0)
        result = session_tf.run(tensor)
        # ...
    ```
    In example above the test_simple is wrapped within graph and session created
    at `session_tf()` fixture. Session and graph are created per each pytest
    function where `session_tf` argument is used.
    """
    with session_context() as session:
        yield session

class session_context(ContextDecorator):
    def __init__(self, graph=None, close_on_exit=True, **kwargs):
        self.graph = graph
        self.close_on_exit = close_on_exit
        self.session = None
        self.session_args = kwargs

    def __enter__(self):
        graph = tf.Graph() if self.graph is None else self.graph
        session = tf.Session(graph=graph, **self.session_args)
        self.session = session
        session.__enter__()
        return session

    def __exit__(self, *exc):
        session = self.session
        session.__exit__(*exc)
        if self.close_on_exit:
            session.close()
        return False
