"""
Discover and orchestrate data-model upgrades.
"""


from ledger import config
from ledger.errors import MigrationError
from ledger.migrate import v1v2
from ledger.migrate.v1v2 import MigrationReport


def discover_v1_accounts() -> list[str]:
    home = config.get_ledger_home()
    if (not home.exists()):
        return []
    names: list[str] = []
    for entry in sorted(home.iterdir()):
        if (entry.is_dir() and entry.name not in {"accounts", "types"}):
            if ((entry / f"{entry.name}.transactions").exists()):
                names.append(entry.name)
    return names


def upgrade(names: list[str] | None = None, force: bool = False) -> list[MigrationReport]:
    targets: list[str] = names if (names is not None and len(names) > 0) else discover_v1_accounts()
    reports: list[MigrationReport] = []
    for name in targets:
        try:
            reports.append(v1v2.migrate_account(name, force = force))
        except MigrationError as error:
            reports.append(MigrationReport(account = name, migrated = False, skipped = False,
                                           count = 0, v1_sum = 0, v2_sum = 0,
                                           message = str(error)))
    return reports
