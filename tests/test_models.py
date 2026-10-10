import datetime

from ledger import models


def test_content_hash_ignores_user_fields() -> None:
    a: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -475,
        statement_memo = "BLUE BOTTLE COFFEE", account = "checking",
        user_memo = "first", category = "Food/Coffee",
    )
    b: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -475,
        statement_memo = "blue  bottle coffee", account = "checking",
        user_memo = "second", category = "Uncategorized",
    )
    assert a.content_hash == b.content_hash   # normalized memo, user fields excluded
    assert a.id != b.id                        # but distinct surrogate identities


def test_different_amount_changes_hash() -> None:
    a: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -475,
        statement_memo = "X", account = "checking",
    )
    b: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -476,
        statement_memo = "X", account = "checking",
    )
    assert a.content_hash != b.content_hash


def test_new_transaction_defaults() -> None:
    txn: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 1, 1), cents = 100, statement_memo = "m", account = "a",
    )
    assert txn.reconciled is False
    assert txn.user_memo == ""
    assert txn.category == ""
    assert len(txn.id) == 32


def test_new_budget_defaults() -> None:
    budget: models.Budget = models.new_budget(
        name = "Laptop", goal_cents = 100000, category = "Income/Paychecks", )
    assert budget.kind is models.BudgetKind.ONE_TIME
    assert budget.account is None
    assert budget.start_date == datetime.date.today()
    assert budget.goal_cents == 100000
    assert len(budget.id) == 32


def test_new_budget_explicit_fields() -> None:
    budget: models.Budget = models.new_budget(
        name = "Car", goal_cents = 500000, category = "Income", account = "checking", start_date = datetime.date(2026, 1, 1),
    )
    assert budget.account == "checking"
    assert budget.start_date == datetime.date(2026, 1, 1)
    assert models.BudgetKind("one_time") is models.BudgetKind.ONE_TIME


def test_new_budget_has_priority_default() -> None:
    budget: models.Budget = models.new_budget(
        name = "Laptop", goal_cents = 100000, category = "Income/Paychecks", )
    assert budget.priority == 0


def test_new_budget_explicit_priority() -> None:
    budget: models.Budget = models.new_budget(
        name = "Car", goal_cents = 500000, category = "Income", priority = 3,
    )
    assert budget.priority == 3


def test_category_cap_fields() -> None:
    cap: models.CategoryCap = models.CategoryCap(category = "Income/Paychecks", draw_bps = 3000)
    assert cap.category == "Income/Paychecks"
    assert cap.draw_bps == 3000


def test_new_budget_has_no_percent_bps() -> None:
    budget: models.Budget = models.new_budget(
        name = "Laptop", goal_cents = 100000, category = "Income/Paychecks",
    )
    assert not hasattr(budget, "percent_bps")
