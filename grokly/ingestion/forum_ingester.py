"""
grokly/ingestion/forum_ingester.py

Exactly 20 curated ERPNext Q&A pairs, stored as separate question and
answer chunks so the knowledge base can match on either.

Categories
----------
HR/Leave Management (4), Expense Claims (4), Payroll (2), Buying (2),
Projects (2), Stock (2), Accounts (2), System (2)

Public API
----------
    from grokly.ingestion.forum_ingester import SEED_QA, run

    chunks_added = run(store, config_loader)
"""

from __future__ import annotations

import hashlib

from grokly.store.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# 20 curated Q&A pairs — (question, answer, category)
# ---------------------------------------------------------------------------

SEED_QA: list[tuple[str, str, str]] = [
    # ── HR / Leave Management (4) ────────────────────────────────────────────
    (
        "How do I submit a leave application in ERPNext?",
        "Go to HR > Leaves > Leave Application > New. Select the Leave Type "
        "(Annual Leave, Sick Leave, etc.), enter From Date and To Date, and add "
        "a Reason. Click Save, then Submit. An email notification is automatically "
        "sent to the Leave Approver assigned in your Employee master. Once the "
        "approver clicks Approve, your leave balance is deducted and the status "
        "changes to Approved.",
        "hr_leave",
    ),
    (
        "What happens if my leave approver is also on leave or unavailable?",
        "If your designated leave approver is on leave or unavailable, an HR "
        "Manager or System Manager can approve the leave application instead. "
        "Go to HR > Leaves > Leave Application, open the pending application, "
        "and click Approve. Alternatively, the Employee master can be updated to "
        "assign a substitute leave approver. You can also escalate using the "
        "Leave Approval Notification workflow to alert a backup approver "
        "automatically after a configurable number of days.",
        "hr_leave",
    ),
    (
        "Where can I check my leave balance in ERPNext?",
        "Employees can check their leave balance in two ways. First, go to "
        "HR > Leaves > Leave Balance Report and filter by Employee and Year. "
        "Second, open any Leave Application form and the available balance for "
        "the selected Leave Type appears automatically in the 'Leave Balance "
        "Before Application' field. HR Managers can view the Leave Allocation "
        "list to see allocations for all employees.",
        "hr_leave",
    ),
    (
        "How do I cancel a submitted leave application in ERPNext?",
        "Open the approved or submitted Leave Application. Click Cancel. The "
        "leave balance is automatically restored. If the leave has already been "
        "reflected in the payroll (e.g., leave without pay deduction), you must "
        "also cancel or amend the relevant salary slip. Note: only the submitter, "
        "Leave Approver, or a user with HR Manager role can cancel a submitted "
        "leave application.",
        "hr_leave",
    ),

    # ── Expense Claims (4) ───────────────────────────────────────────────────
    (
        "How do I submit an expense claim in ERPNext?",
        "Go to HR > Expenses > Expense Claim > New. Select the Employee name "
        "and choose an Expense Approver. In the Expenses table, add one row per "
        "expense: set Expense Date, Expense Type, Description, and Amount. "
        "Attach scanned receipts using the Attach button. Click Save, then "
        "Submit. The expense approver receives an email notification to review "
        "and approve the claim. Once approved, a payment entry reimburses the "
        "employee.",
        "expense",
    ),
    (
        "What happens after an expense claim is submitted in ERPNext?",
        "After submission the expense claim status changes to Submitted and the "
        "designated Expense Approver is notified by email. The approver opens "
        "the claim, reviews each expense line, and clicks Approve (or Reject "
        "with a reason). Once approved, an Accounts Payable entry is created. "
        "To reimburse the employee, go to Accounts > Payment Entry > New, set "
        "Party Type to Employee, select the employee, and link the approved "
        "expense claim. Submit the payment to complete reimbursement.",
        "expense",
    ),
    (
        "How do I handle foreign currency expenses in an ERPNext expense claim?",
        "When entering an expense line in a foreign currency, set the Currency "
        "field to the foreign currency (e.g., USD) and enter the amount in that "
        "currency. ERPNext automatically fetches the exchange rate using the "
        "Currency Exchange master or the live rate if enabled. The system "
        "converts the amount to your company's base currency for accounting. "
        "You can override the exchange rate manually in the expense line if the "
        "actual rate on your receipt differs.",
        "expense",
    ),
    (
        "How do expense claim approval limits work in ERPNext?",
        "ERPNext lets you define maximum claim amounts per Expense Approver. "
        "Go to HR > Setup > Expense Claim Type and set a maximum claim amount "
        "if needed. Additionally, in the Employee master you can set a specific "
        "Expense Approver. If a claim exceeds the approver's authorised limit, "
        "it can be escalated to a senior approver or HR Manager. Workflow rules "
        "can be configured under Setup > Workflow to route high-value claims to "
        "additional approval levels automatically.",
        "expense",
    ),

    # ── Payroll (2) ──────────────────────────────────────────────────────────
    (
        "How do I view my payslip in ERPNext?",
        "Employees can view their payslip by going to HR > Payroll > Salary "
        "Slip and filtering by their Employee ID and the relevant month. If the "
        "Employee Self Service portal is enabled, employees can log in and "
        "navigate to My Payslips to view and download PDF copies of all their "
        "salary slips directly without requiring HR access.",
        "payroll",
    ),
    (
        "How is overtime calculated in ERPNext payroll?",
        "Overtime in ERPNext is typically handled through a custom Salary "
        "Component with a formula. Create a component named 'Overtime' under "
        "Payroll > Salary Component. Set the formula to multiply overtime hours "
        "by the hourly rate, for example: (base / 26 / 8) * overtime_hours * "
        "1.5 for time-and-a-half. Link this component in the Salary Structure "
        "under Earnings. Overtime hours can be fed in via Additional Salary or "
        "by using a custom field on the Salary Slip to capture the hours before "
        "payroll is processed.",
        "payroll",
    ),

    # ── Buying (2) ───────────────────────────────────────────────────────────
    (
        "How do I raise a Purchase Order in ERPNext?",
        "Go to Buying > Purchase Order > New. Select the Supplier and set the "
        "Required By date. Add items in the Items table with Item Code, "
        "Quantity, Rate, and Warehouse. Review taxes in the Taxes and Charges "
        "section. Save the draft, then Submit to confirm the order. The PO "
        "status changes to 'To Receive and Bill'. Approval workflow can be "
        "configured so that high-value purchase orders require a manager to "
        "approve before submission. Once approved the supplier can be notified "
        "by email directly from the PO.",
        "buying",
    ),
    (
        "What is the difference between a Purchase Order and a Material Request "
        "in ERPNext?",
        "A Material Request is an internal document raised by a department or "
        "warehouse to signal that stock is needed — it does not involve an "
        "external supplier. A Purchase Order is a formal legal document sent to "
        "a supplier to buy specific items at agreed prices and delivery dates. "
        "The typical flow is: Material Request > Request for Quotation > "
        "Supplier Quotation > Purchase Order > Purchase Receipt > Purchase "
        "Invoice. A Material Request can automatically trigger the creation of "
        "a Purchase Order through the 'Create Purchase Order' button if the "
        "item's default supplier is configured.",
        "buying",
    ),

    # ── Projects (2) ─────────────────────────────────────────────────────────
    (
        "How do I log time against a project in ERPNext?",
        "Go to Projects > Timesheets > New Timesheet. Select the Employee. "
        "In the Time Logs table, add a row and set Activity Type, Project, "
        "Task (optional), From Time, and To Time. The Hours field is calculated "
        "automatically. Save and Submit the timesheet. The logged hours appear "
        "in the project's Actual Time field. For billable projects, submitted "
        "timesheets can be used to generate a Sales Invoice via the 'Make "
        "Sales Invoice' button on the project.",
        "projects",
    ),
    (
        "What is the difference between a Project and a Task in ERPNext?",
        "A Project is the top-level container — it has a budget, timeline, "
        "customer, and overall status. Tasks are individual work items nested "
        "inside a project. Each task can be assigned to a specific employee, "
        "given a priority (Low/Medium/High/Urgent), estimated hours, and "
        "a status (Open/Working/Pending Review/Completed/Cancelled). Time logs "
        "and expenses are recorded at the task level and roll up to the project. "
        "You can visualise tasks on a Gantt chart from the project form.",
        "projects",
    ),

    # ── Stock (2) ────────────────────────────────────────────────────────────
    (
        "How do I transfer stock between warehouses in ERPNext?",
        "Go to Stock > Stock Transactions > Stock Entry > New. Set the Purpose "
        "to 'Material Transfer'. In the Items table, add each item with the "
        "Source Warehouse and Target Warehouse. Enter the Quantity to transfer. "
        "Save and Submit. The stock ledger is updated immediately — stock is "
        "deducted from the source warehouse and added to the target warehouse. "
        "You can also initiate a transfer directly from a Material Request by "
        "selecting 'Transfer' as the purpose.",
        "stock",
    ),
    (
        "How do I check current stock levels in ERPNext?",
        "Go to Stock > Reports > Stock Balance to see the current quantity and "
        "valuation for all items across all warehouses. Filter by Item, "
        "Warehouse, or Item Group. For a single item, open the Item master and "
        "click the 'Stock Ledger' button to see all movements. You can also "
        "use Stock > Reports > Itemwise Recommended Reorder Level to identify "
        "items that need replenishment based on reorder levels.",
        "stock",
    ),

    # ── Accounts (2) ────────────────────────────────────────────────────────
    (
        "What is the difference between a Sales Invoice and a Delivery Note "
        "in ERPNext?",
        "A Delivery Note records the physical movement of goods from your "
        "warehouse to the customer — it updates stock levels but does not post "
        "any accounting entry by default. A Sales Invoice is the financial "
        "document that creates the accounts receivable entry and records "
        "revenue. The typical flow is: Sales Order > Delivery Note > Sales "
        "Invoice. Alternatively, you can bill before delivery (Sales Order > "
        "Sales Invoice > Delivery Note). ERPNext links them so stock and "
        "accounting remain consistent.",
        "accounts",
    ),
    (
        "How do I do bank reconciliation in ERPNext?",
        "Go to Accounts > Banking and Payments > Bank Reconciliation Statement. "
        "Select the Bank Account and the date range. You can upload a bank "
        "statement CSV using the Upload Bank Statement button. ERPNext tries "
        "to auto-match transactions to existing Payment Entries or Journal "
        "Entries. For each unmatched bank transaction, click Match to link it "
        "manually, or Create Entry to record a new payment or journal. All "
        "matched items move to the reconciled list. Outstanding items indicate "
        "missing entries.",
        "accounts",
    ),

    # ── System (2) ──────────────────────────────────────────────────────────
    (
        "How do I reset my ERPNext password?",
        "On the ERPNext login page click 'Forgot Password' and enter your "
        "registered email address. A password reset link is emailed to you. "
        "Click the link and enter a new password. If you cannot access your "
        "email, ask an Administrator to go to Setup > Users, open your user "
        "record, enter a new password in the 'New Password' field, and save. "
        "For administrator password recovery when email is unavailable, use "
        "the bench command: bench --site <site-name> set-admin-password "
        "<new-password> from the server terminal.",
        "system",
    ),
    (
        "How do I export ERPNext data to Excel?",
        "Open any list view (e.g., Sales Invoice list). Apply filters as "
        "needed to narrow the records. Click the Menu button (three dots) in "
        "the top-right corner and select 'Export'. Choose 'Excel' as the "
        "format. Select which columns to include and click Export. The file "
        "downloads as an XLSX file. For large datasets use the Data Export "
        "tool under Setup > Setup > Data > Export Data, which lets you export "
        "entire DocTypes including all fields in one go.",
        "system",
    ),
]

assert len(SEED_QA) == 20, f"Expected 20 SEED_QA pairs, got {len(SEED_QA)}"


# ---------------------------------------------------------------------------
# Ingester entry point
# ---------------------------------------------------------------------------


def run(store: ChromaStore, config_loader=None) -> int:
    """
    Ingest all SEED_QA pairs into *store* as separate question and answer chunks.

    Parameters
    ----------
    store : ChromaStore
        Destination knowledge base.
    config_loader : ConfigLoader, optional
        If provided, the forum source enabled flag in sources_qna.json is checked.

    Returns
    -------
    int
        Total number of chunks added or updated.
    """
    if config_loader is not None:
        sources = config_loader.qna.get("sources", [])
        enabled = [s for s in sources if s.get("enabled", True)]
        if not enabled:
            print("  Forum source is disabled in sources_qna.json — skipping.")
            return 0

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for question, answer, category in SEED_QA:
        q_id = hashlib.md5(f"q:{question}".encode("utf-8")).hexdigest()
        texts.append(question)
        metadatas.append(
            {
                "source": "forum",
                "file_type": "forum_qa",
                "category": category,
                "chunk_type": "question",
            }
        )
        ids.append(q_id)

        a_id = hashlib.md5(f"a:{answer}".encode("utf-8")).hexdigest()
        texts.append(answer)
        metadatas.append(
            {
                "source": "forum",
                "file_type": "forum_qa",
                "category": category,
                "chunk_type": "answer",
            }
        )
        ids.append(a_id)

    return store.upsert(texts=texts, metadatas=metadatas, ids=ids)
