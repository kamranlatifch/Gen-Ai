"""Step 1 of the multimodal RAG pipeline: build a test PDF.

report.md (in RAG/) was a synthetic text document, purpose-built so we knew
the right answers ahead of time. This is the same idea for charts: six pages,
each one clear matplotlib chart with a specific, unambiguous trend, plus a
GROUND_TRUTH dict recording which page shows what - so later, when we grade
retrieval, we can check "did it fetch the right page" with a plain code
comparison instead of guessing.

Run: python make_charts_pdf.py
"""

import matplotlib

matplotlib.use("Agg")  # no display needed, just render to the PDF file
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PDF_PATH = "charts.pdf"

# page number (1-indexed, matches PDF page order) -> what a correct answer
# needs to say. Used later purely for grading, never shown to the model.
GROUND_TRUTH = {
    1: "Revenue trended upward across all 8 quarters, from $2.1M to $5.8M.",
    2: "Latency trended downward over the year, with a sharp drop in July "
    "(the month a caching layer shipped).",
    3: "Churn spiked in December, roughly triple every other month's rate.",
    4: "Monthly active users were flat around 40k, then jumped to ~95k in "
    "September (the month of a product launch) and stayed there.",
    5: "Website traffic is seasonal: it peaks in November-December and dips "
    "every summer (June-August).",
    6: "Defect rate declined after May, when a QA program started; it fell "
    "from around 4% to under 1%.",
}


def build_pdf():
    with PdfPages(PDF_PATH) as pdf:
        # --- Page 1: revenue, clear upward trend ---
        fig, ax = plt.subplots(figsize=(8, 5))
        quarters = ["23Q1", "23Q2", "23Q3", "23Q4", "24Q1", "24Q2", "24Q3", "24Q4"]
        revenue = [2.1, 2.6, 3.0, 3.4, 4.0, 4.5, 5.1, 5.8]
        ax.plot(quarters, revenue, marker="o", linewidth=2)
        ax.set_title("Quarterly Revenue ($M)")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Revenue ($M)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: latency, downward trend with a step-change in July ---
        fig, ax = plt.subplots(figsize=(8, 5))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        latency = [420, 410, 400, 395, 390, 385, 210, 200, 195, 190, 185, 180]
        ax.plot(months, latency, marker="o", color="darkred", linewidth=2)
        ax.axvline(x=6, color="gray", linestyle="--", alpha=0.6)
        ax.text(6.1, 380, "caching layer shipped", fontsize=9, color="gray")
        ax.set_title("Server Response Latency (ms)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: churn, December spike ---
        fig, ax = plt.subplots(figsize=(8, 5))
        churn = [2.1, 2.0, 2.2, 2.1, 2.3, 2.0, 2.2, 2.1, 2.3, 2.2, 2.4, 6.8]
        ax.bar(months, churn, color="orange")
        ax.set_title("Monthly Customer Churn Rate (%)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Churn Rate (%)")
        ax.grid(True, alpha=0.3, axis="y")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: MAU, flat then step-jump in September ---
        fig, ax = plt.subplots(figsize=(8, 5))
        mau = [41, 39, 40, 42, 40, 41, 40, 39, 95, 96, 94, 97]
        ax.plot(months, mau, marker="s", color="seagreen", linewidth=2)
        ax.axvline(x=8, color="gray", linestyle="--", alpha=0.6)
        ax.text(8.1, 55, "product launch", fontsize=9, color="gray")
        ax.set_title("Monthly Active Users (thousands)")
        ax.set_xlabel("Month")
        ax.set_ylabel("MAU (thousands)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 5: website traffic, seasonal (peaks Nov-Dec, dips summer) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        traffic = [80, 78, 82, 85, 90, 60, 55, 58, 88, 95, 130, 140]
        ax.plot(months, traffic, marker="o", color="purple", linewidth=2)
        ax.fill_between(range(len(months)), traffic, alpha=0.15, color="purple")
        ax.set_title("Website Traffic (thousands of visits)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Visits (thousands)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 6: defect rate, decline after QA program (May) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        defects = [4.1, 4.0, 3.9, 4.2, 4.0, 2.8, 2.1, 1.6, 1.2, 0.9, 0.8, 0.7]
        ax.plot(months, defects, marker="o", color="crimson", linewidth=2)
        ax.axvline(x=4, color="gray", linestyle="--", alpha=0.6)
        ax.text(4.1, 3.7, "QA program started", fontsize=9, color="gray")
        ax.set_title("Manufacturing Defect Rate (%)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Defect Rate (%)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {PDF_PATH} with {len(GROUND_TRUTH)} pages:")
    for page, truth in GROUND_TRUTH.items():
        print(f"  page {page}: {truth}")


if __name__ == "__main__":
    build_pdf()
