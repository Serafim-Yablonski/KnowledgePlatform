# Import submodules so Celery's include=["src.workers.tasks"] discovers tasks on
# worker startup. Using submodule imports (not from-imports) avoids shadowing the
# submodule names in the package namespace, which breaks monkeypatching in tests.
import src.workers.tasks.embed_chunks  # noqa: F401
import src.workers.tasks.extract_text  # noqa: F401
