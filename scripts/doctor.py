from __future__ import annotations

from godmode.util.doctor import print_doctor, run_doctor


def main() -> int:
    checks = run_doctor()
    return print_doctor(checks)


if __name__ == "__main__":
    raise SystemExit(main())


