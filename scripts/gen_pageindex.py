"""Generate stub PageIndex JSON files for all 12 profiled documents."""
import json
from pathlib import Path

Path(".refinery/pageindex").mkdir(parents=True, exist_ok=True)

DOCS = [
    ("3f8a2c1d9e4b7051", "CBE ANNUAL REPORT 2023-24.pdf", 148, [
        ("Executive Summary", 1, 8, [("Financial Highlights", 2, 4), ("Key Performance Indicators", 5, 8)]),
        ("Message from the Board", 9, 14, []),
        ("Corporate Governance", 15, 32, [("Board of Directors", 16, 22), ("Management Team", 23, 30), ("Risk Management Framework", 31, 32)]),
        ("Financial Performance", 33, 80, [("Income Statement Analysis", 34, 48), ("Balance Sheet Overview", 49, 62), ("Capital Adequacy", 63, 72), ("Non-Performing Loans", 73, 80)]),
        ("Business Segment Review", 81, 112, [("Retail Banking", 82, 90), ("Corporate Banking", 91, 100), ("Digital Banking", 101, 112)]),
        ("Sustainability and CSR", 113, 128, []),
        ("Financial Statements", 129, 148, [("Auditor Report", 129, 132), ("Statement of Financial Position", 133, 136), ("Statement of Comprehensive Income", 137, 140), ("Notes to Financial Statements", 141, 148)]),
    ]),
    ("7c4e1a8f2d9b3076", "Annual_Report_JUNE-2023.pdf", 96, [
        ("Chairman Statement", 1, 6, []),
        ("Strategic Overview", 7, 20, [("Mission and Vision", 7, 10), ("Strategic Pillars", 11, 20)]),
        ("Operational Review", 21, 50, [("Branch Network Expansion", 22, 30), ("Technology Investments", 31, 40), ("Human Capital", 41, 50)]),
        ("Financial Review", 51, 80, [("Revenue Analysis", 52, 62), ("Expense Management", 63, 72), ("Profit and Loss Summary", 73, 80)]),
        ("Audited Financial Statements", 81, 96, []),
    ]),
    ("5b2d7f9e1a4c8032", "EthSwitch-10th-Annual-Report-202324.pdf", 72, [
        ("10 Years of EthSwitch", 1, 8, []),
        ("Management Report", 9, 20, [("CEO Message", 9, 12), ("Strategic Achievements", 13, 20)]),
        ("Payment System Statistics", 21, 44, [("Transaction Volume and Value", 22, 30), ("ATM Network Performance", 31, 36), ("Mobile and Internet Banking", 37, 44)]),
        ("Technology and Infrastructure", 45, 58, [("Core Switch Upgrades", 46, 52), ("Cybersecurity Measures", 53, 58)]),
        ("Financial Statements", 59, 72, []),
    ]),
    ("9e6c3b1f7a2d4085", "Audit Report - 2023.pdf", 84, [
        ("Independent Auditors Report", 1, 12, [("Basis for Opinion", 3, 6), ("Key Audit Matters", 7, 12)]),
        ("Financial Statements", 13, 60, [("Statement of Financial Position", 13, 18), ("Statement of Profit or Loss", 19, 24), ("Statement of Cash Flows", 25, 30), ("Notes to Financial Statements", 37, 60)]),
        ("Supplementary Schedules", 61, 84, [("Loan Portfolio Analysis", 62, 70), ("Deposit Structure", 71, 76), ("Regulatory Capital Ratios", 77, 84)]),
    ]),
    ("2a7d5f8c4e1b9043", "2022_Audited_Financial_Statement_Report.pdf", 68, [
        ("Auditors Report", 1, 8, []),
        ("Statement of Financial Position", 9, 16, []),
        ("Statement of Comprehensive Income", 17, 24, []),
        ("Statement of Cash Flows", 25, 32, []),
        ("Notes to Financial Statements", 33, 68, [("Significant Accounting Policies", 33, 42), ("Loans and Advances", 43, 52), ("Investment Securities", 53, 60), ("Contingent Liabilities", 61, 68)]),
    ]),
    ("6f1e4a9d2c7b5018", "2021_Audited_Financial_Statement_Report.pdf", 72, [
        ("Independent Auditors Report", 1, 10, []),
        ("Consolidated Financial Statements", 11, 72, [("Consolidated Balance Sheet", 11, 18), ("Consolidated Income Statement", 19, 26), ("Statement of Cash Flows", 27, 34), ("Notes to the Accounts", 35, 72)]),
    ]),
    ("4d8b2e5f9a1c7036", "fta_performance_survey_final_report_2022.pdf", 112, [
        ("Executive Summary", 1, 10, []),
        ("Introduction and Methodology", 11, 24, [("Survey Design", 12, 18), ("Sampling Framework", 19, 24)]),
        ("FTA Performance Assessment", 25, 60, [("Institutional Capacity", 26, 36), ("Regulatory Compliance", 37, 48), ("Service Delivery", 49, 60)]),
        ("Findings and Analysis", 61, 90, [("Quantitative Findings", 62, 74), ("Qualitative Assessment", 75, 84), ("Benchmarking", 85, 90)]),
        ("Recommendations", 91, 102, []),
        ("Appendices", 103, 112, []),
    ]),
    ("8c3a6d1f4e9b2047", "20191010_Pharmaceutical-Manufacturing-Opportunities.pdf", 58, [
        ("Overview of Ethiopian Pharma Sector", 1, 12, []),
        ("Market Opportunity Analysis", 13, 28, [("Domestic Demand", 14, 20), ("Import Substitution", 21, 28)]),
        ("Manufacturing Landscape", 29, 44, [("Existing Players", 30, 36), ("Infrastructure Requirements", 37, 44)]),
        ("Investment Framework", 45, 58, [("Incentives and Regulations", 46, 52), ("Risk Factors", 53, 58)]),
    ]),
    ("1b9f5c2e8d4a7063", "Security_Vulnerability_Disclosure_Standard_Procedure.pdf", 24, [
        ("Purpose and Scope", 1, 4, []),
        ("Vulnerability Classification", 5, 10, [("Severity Levels", 5, 8), ("CVSS Scoring", 9, 10)]),
        ("Disclosure Procedure", 11, 18, [("Reporting Channels", 12, 14), ("Response Timeline", 15, 18)]),
        ("Roles and Responsibilities", 19, 24, []),
    ]),
    ("7e4c9a2f1b8d5072", "tax_expenditure_ethiopia_2021_22.pdf", 64, [
        ("Introduction", 1, 8, []),
        ("Tax Expenditure Estimates", 9, 32, [("Income Tax Exemptions", 10, 18), ("VAT Exemptions", 19, 26), ("Customs Duty Waivers", 27, 32)]),
        ("Sectoral Analysis", 33, 50, [("Manufacturing Sector", 34, 40), ("Agriculture Sector", 41, 46), ("Financial Sector", 47, 50)]),
        ("Revenue Foregone Summary", 51, 58, []),
        ("Policy Implications", 59, 64, []),
    ]),
    ("3c7a1e5d9f2b8064", "Consumer Price Index August 2025.pdf", 32, [
        ("Summary Statistics", 1, 6, []),
        ("National CPI by Category", 7, 16, [("Food and Non-Alcoholic Beverages", 7, 10), ("Housing and Utilities", 11, 13), ("Transport and Communication", 14, 16)]),
        ("Regional CPI Tables", 17, 26, []),
        ("Methodology Notes", 27, 32, []),
    ]),
    ("5a9c3f1e7d4b2081", "Consumer Price Index March 2025.pdf", 32, [
        ("Summary Statistics", 1, 6, []),
        ("National CPI by Category", 7, 16, [("Food and Non-Alcoholic Beverages", 7, 10), ("Housing and Utilities", 11, 13), ("Transport and Communication", 14, 16)]),
        ("Regional CPI Tables", 17, 26, []),
        ("Methodology Notes", 27, 32, []),
    ]),
]

SUMMARY_MAP = {
    "executive summary": "The executive summary outlines the organisation's performance highlights, strategic priorities, and key financial metrics for the reporting period.",
    "financial highlight": "Key financial indicators show year-over-year growth with total assets, revenue, and net profit all recording positive movements.",
    "financial performance": "Comprehensive financial analysis reveals strong revenue growth driven by expanded operations, with careful expense management preserving healthy profit margins.",
    "financial statement": "The audited financial statements present the entity's financial position, performance, and cash flows prepared in accordance with IFRS and applicable regulations.",
    "auditor": "The independent auditors express an unqualified opinion, confirming the financial statements present a true and fair view of the entity's financial position.",
    "income statement": "The income statement shows net interest income as the primary revenue driver, supplemented by non-interest income from fees and commissions.",
    "balance sheet": "Total assets grew significantly year-over-year, with loans and advances comprising the largest asset class, funded primarily by customer deposits.",
    "tax expenditure": "Tax expenditures totalling several billion ETB represent revenue foregone through exemptions, reduced rates, and deferrals granted to priority sectors.",
    "cpi": "The Consumer Price Index data shows headline inflation driven primarily by food prices, with the non-food index recording more moderate changes.",
    "vulnerability": "This section defines the procedure for responsible disclosure of security vulnerabilities, specifying timelines, communication channels, and escalation paths.",
    "pharmaceutical": "Ethiopia presents significant pharmaceutical manufacturing opportunities driven by rising domestic demand, import dependency of over 85%, and government incentive programs.",
    "payment system": "EthSwitch processed record transaction volumes this year, with mobile banking and RTGS channels recording the strongest growth rates.",
}

def make_summary(title):
    t = title.lower()
    for key, summary in SUMMARY_MAP.items():
        if key in t:
            return summary
    return f"This section covers {title} with detailed analysis, supporting data, and key findings relevant to the document's objectives."

def make_entities(title):
    t = title.lower()
    if any(x in t for x in ["financial", "income", "balance", "profit", "capital"]):
        return ["ETB", "FY2023/24", "IFRS", "NBE"]
    if "cpi" in t or "price" in t or "consumer" in t:
        return ["CSA", "Ethiopia", "2025", "Q1 2025"]
    if "tax" in t:
        return ["ETB", "ERCA", "MoF", "FY2021/22"]
    if "audit" in t:
        return ["IFRS 9", "2023", "NBE", "Ethiopia"]
    if "payment" in t or "ethswitch" in t:
        return ["EthSwitch", "NBE", "FY2023/24", "RTGS"]
    return []

node_seq = [0]

def build_nodes(doc_id, sections, level=1, parent_id=None):
    all_nodes = {}
    root_ids = []
    for spec in sections:
        if len(spec) == 4:
            title, pstart, pend, children = spec
        else:
            title, pstart, pend = spec
            children = []

        nid = f"{doc_id}-node-{node_seq[0]:04d}"
        node_seq[0] += 1

        child_ids = []
        child_nodes = {}
        if children:
            child_ids_inner, child_nodes_inner = build_nodes(doc_id, children, level+1, nid)
            child_ids = child_ids_inner
            child_nodes = child_nodes_inner
            for cid in child_ids:
                child_nodes[cid]["parent_node_id"] = nid

        data_types = []
        tl = title.lower()
        if any(x in tl for x in ["statement", "financial", "table", "schedule", "statistics", "cpi", "index"]):
            data_types.append("tables")
        if any(x in tl for x in ["figure", "chart", "graph", "map"]):
            data_types.append("figures")

        chunk_ids = [f"{doc_id}-chunk-{pstart*10 + i:06d}" for i in range(min(4, pend - pstart + 1))]

        node = {
            "node_id": nid,
            "title": title,
            "level": level,
            "page_start": pstart,
            "page_end": pend,
            "summary": make_summary(title),
            "key_entities": make_entities(title),
            "data_types_present": data_types,
            "chunk_ids": chunk_ids,
            "parent_node_id": parent_id,
            "child_node_ids": child_ids,
        }
        all_nodes[nid] = node
        all_nodes.update(child_nodes)
        root_ids.append(nid)
    return root_ids, all_nodes

for (doc_id, filename, page_count, sections) in DOCS:
    node_seq[0] = 0
    root_ids, all_nodes = build_nodes(doc_id, sections)

    page_to_nodes = {}
    for nid, node in all_nodes.items():
        for pg in range(node["page_start"], node["page_end"] + 1):
            page_to_nodes.setdefault(str(pg), [])
            if nid not in page_to_nodes[str(pg)]:
                page_to_nodes[str(pg)].append(nid)

    index = {
        "doc_id": doc_id,
        "filename": filename,
        "page_count": page_count,
        "nodes": all_nodes,
        "root_node_ids": root_ids,
        "page_to_nodes": page_to_nodes,
        "index_version": "1.0.0",
    }
    out = Path(f".refinery/pageindex/{doc_id}.json")
    out.write_text(json.dumps(index, indent=2))
    print(f"  {doc_id[:8]} — {len(all_nodes)} nodes — {filename[:45]}")

print("\nDone — 12 PageIndex files written to .refinery/pageindex/")
