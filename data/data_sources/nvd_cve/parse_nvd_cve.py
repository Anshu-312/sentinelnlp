import os
import json

def parse_cve_feed(raw_file):
    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cve_items = []
    for item in data.get("CVE_Items", []):
        cve = {
            "cve_id": item["cve"]["CVE_data_meta"]["ID"],
            "description": item["cve"]["description"]["description_data"][0]["value"],
            "published_date": item.get("publishedDate", ""),
            "last_modified_date": item.get("lastModifiedDate", ""),
            "cvss_score": item.get("impact", {}).get("baseMetricV3", {}).get("cvssV3", {}).get("baseScore", None),
            "severity": item.get("impact", {}).get("baseMetricV3", {}).get("cvssV3", {}).get("baseSeverity", None),
            "affected_products": []
        }

        # Parse affected products
        nodes = item.get("configurations", {}).get("nodes", [])
        for node in nodes:
            for cpe in node.get("cpe_match", []):
                cve["affected_products"].append(cpe.get("cpe23Uri", ""))

        cve_items.append(cve)

    return cve_items

def save_parsed_cves(processed_file, cve_items):
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(cve_items, f, indent=2)

if __name__ == "__main__":
    raw_file = r"C:\Users\Anshu Bhadani\OneDrive\Desktop\sentinelnlp\data\data_sources\nvd_cve\raw\nvdcve-1.1-modified.json"
    processed_file = "data_sources/nvd_cve/processed/nvd_recent_cves.json"
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)

    cve_data = parse_cve_feed(raw_file)
    save_parsed_cves(processed_file, cve_data)
    print(f"Parsed {len(cve_data)} CVEs from NVD feed!")
