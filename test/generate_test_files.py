import os

test_samples = {
    "sample1_basic.txt": """APT29 used Mimikatz to escalate privileges via DLL Sideloading.
The malware Emotet exploited CVE-2021-34527 to gain access to Microsoft systems.""",

    "sample2_lateral_movement.txt": """FIN7 actors exploited CVE-2023-12345 using Cobalt Strike to move laterally inside corporate networks.""",

    "sample3_powergrid.txt": """Sandworm used BlackEnergy malware to disable Ukraine's power grid.""",

    "sample4_obfuscated.txt": """APT28 leveraged a custom loader disguised as rundll32.exe to deploy Zebrocy malware.
They utilized a backdoor over HTTPS with DNS fallback for C2 communication.""",

    "sample5_multiactor.txt": """Both APT41 and Mustang Panda exploited the same vulnerability CVE-2023-5678 to compromise education sector networks.""",

    "sample6_noisy.txt": """(!! Malware Alert !!) AgentTesla used in phishing attacks targeting finance (banking) firms!!! Ref: https://badurl.com/alert123""",

    "sample7_mixed_lang.txt": """Lazarus Group 사용한 RAT는 Windows 시스템을 감염시킵니다. Tool 이름은 "Dtrack"입니다.""",

    "sample8_nested_context.txt": """UNC2452 first used PowerShell Empire, and after persistence, deployed SUNBURST malware through SolarWinds update mechanism.""",

    "sample9_infra_targets.txt": """APT10 used PlugX to target servers at IP 203.0.113.5 hosting government databases.""",

    "sample10_multiple_tools.txt": """Turla Group deployed both Kazuar and Gazer malware families to maintain long-term access within defense contractors."""
}

# Create output directory
output_dir = "test_samples"
os.makedirs(output_dir, exist_ok=True)

# Write files
for filename, content in test_samples.items():
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

print(f"[✓] Generated {len(test_samples)} test files in '{output_dir}/'")
