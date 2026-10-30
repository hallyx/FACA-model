"""Compatibility command; the primary example is examples/inference.py."""
if __package__:
    from .inference import main, parse_args
else:
    from inference import main, parse_args

if __name__ == '__main__':
    main()
