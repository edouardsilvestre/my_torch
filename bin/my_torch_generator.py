#!/usr/bin/env python3
import sys

try:
    from my_torch.generator import main
except ImportError as error:
    print(f"my_torch_generator: {error}", file=sys.stderr)
    sys.exit(84)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
