from fedlab_core.flower import app as core

app = core.client  # Flower finds `app` by reading this file, not by importing it
