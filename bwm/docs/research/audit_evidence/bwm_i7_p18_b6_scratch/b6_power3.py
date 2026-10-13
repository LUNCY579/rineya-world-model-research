exec(open("b6_power2.py").read().split("import sys")[0])
print("P1 S-direct n=150 (P(p<.05 & b>c) | P(MET: +10pp & p<.05))")
for lab,p10,p01 in [("+8pp (r1 11-15 analog .095/.014)",.095,.014),("+10pp (.13/.03)",.13,.03),("+13pp (.17/.04)",.17,.04),("+15pp (.18/.03)",.18,.03),("+22pp (r1 6-10 analog .306/.083)",.306,.083)]:
    print(f"  {lab}: {power(150,p10,p01,.05):.2f} | {power(150,p10,p01,.05,10):.2f}")
print("P2 S-beam (n=64 | n=80 | n=100): P(MET +10pp & p<.05)")
for lab,p10,p01 in [("H2-vs-B2 at 12-15 analog +16pp (.226/.065)",.226,.065),("+12pp (.17/.05)",.17,.05),("+12pp noisy (.20/.08)",.20,.08),("+8pp (.14/.06)",.14,.06),("null 25% discordance (.125/.125)",.125,.125)]:
    print(f"  {lab}: " + " | ".join(f"{power(n,p10,p01,.05,10):.2f}" for n in (64,80,100)))
