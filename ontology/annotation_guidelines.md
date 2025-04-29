# Annotation Guidelines v1.0 — Cyber Threat Intelligence

These guidelines are for annotating entities and relationships in Cyber Threat Intelligence (CTI) reports. The goal is to create high-quality training data for Named Entity Recognition (NER) and Relation Extraction.

---

## 🧠 Entity Types

| Entity Type    | Examples                             | Description                                         |
|----------------|--------------------------------------|-----------------------------------------------------|
| THREAT_ACTOR   | APT28, Lazarus Group                 | Known attacker groups or hacker collectives        |
| MALWARE        | Emotet, WannaCry                     | Malware families or specific threats               |
| TOOL           | Mimikatz, Cobalt Strike              | Tools used during attacks (open source or illicit) |
| VULNERABILITY  | CVE-2021-34527, BlueKeep             | Vulnerabilities exploited by attackers             |
| TACTIC         | Initial Access, Lateral Movement     | MITRE ATT&CK Tactic category                       |
| TECHNIQUE      | Spearphishing Link, DLL Sideloading  | MITRE ATT&CK Techniques                            |
| TARGET         | Microsoft, supply chain, admin user  | Victims or systems being targeted                  |
| INFRASTRUCTURE | IP addresses, domains, hosting VPS   | Attacker-controlled servers or platforms           |

---

## 🔗 Relation Types

| Relation       | From            | To                        | Example                                   |
|----------------|-----------------|---------------------------|-------------------------------------------|
| uses           | THREAT_ACTOR    | MALWARE / TOOL            | APT29 → uses → Cobalt Strike              |
| exploits       | MALWARE         | VULNERABILITY             | WannaCry → exploits → CVE-2017-0144       |
| executes       | THREAT_ACTOR    | TECHNIQUE                 | Lazarus → executes → Spearphishing Link   |
| part_of        | TECHNIQUE       | TACTIC                    | DLL Sideloading → part_of → Execution     |
| targets        | THREAT_ACTOR    | TARGET                    | APT28 → targets → Defense Contractor       |
| hosts          | INFRASTRUCTURE  | TOOL / MALWARE            | 123.45.67.89 → hosts → Mimikatz           |
| facilitates    | TOOL            | TECHNIQUE                 | Empire → facilitates → Command Execution  |

---

## ✅ Annotation Rules

- Annotate **whole entity names** (e.g., “APT29 Group” not just “APT29”).
- Annotate **first valid instance** in each paragraph or sentence.
- Use **CVE IDs** when mentioned (CVE-2020-0601, etc.).
- Link TECHNIQUES only to correct TACTICS based on MITRE ATT&CK.
- Annotate IPs/Domains **only if** clearly related to attacker infrastructure.
- Avoid casual references unless tied to an attack context.
- Do not annotate overlapping entities unless nested meaningfully.

---

## 📦 Output Format Example (for annotation)

```json
{
  "text": "APT29 used Mimikatz to escalate privileges via DLL Sideloading.",
  "entities": [
    {"start": 0, "end": 5, "label": "THREAT_ACTOR"},
    {"start": 11, "end": 19, "label": "TOOL"},
    {"start": 47, "end": 64, "label": "TECHNIQUE"}
  ],
  "relations": [
    {"source": 0, "target": 1, "type": "uses"},
    {"source": 1, "target": 2, "type": "facilitates"}
  ]
}