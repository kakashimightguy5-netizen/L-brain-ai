# Validation diagnosis: stronger sensory models

[[42 - Stronger Architecture and Evidence Protocol]]

The first exploratory seed442 showed CNN noisy validation65.74%vsMLP73.82%. A hard clean-accuracy guard can prevent robust adaptation entirely; early spatial pooling can also remove useful8×8details. These validation observations prompted the following registered revision before new fits. The first checkpoint/script/source remain archived as diagnostics; no success claim is based on that interrupted run.

Use fresh seeds642–646. Retain full8×8resolution through three16channelconvolutions and a32featurelinearreadout; approximately38kparameters, matched toflatMLPwidth162. Remove the clean guard from robust-branch training. Preserve the clean pretrained expert separately. Train an offline condition critic on TRAINING clean/noisy examples, using clean-expert features, both classprobabilities and pixelmean/std/nonzero fraction. No handcrafted exactquantization-grid detector. The critic chooses clean vsrobustexpert; settings selected on independent validation withcleanqualitywithin.5pointofthecleanexpert andhighestnoisyaccuracy. This doublesexpertstorage/inference and receives offlinecorruption-condition supervision; it is not intrinsic truth discovery.

Use the robust expert alone for accumulating independent observations; the condition critic sees one-observationstyle shifts and might not transfer to averages. Report both rawexperts, routedexperts and memorycorrector. Keep original equal-budgetaugmentation and all distribution tests from42.

Execution note: an initial checkpoint included a NumPy scalar calibration threshold, which the strict weights-only loader rejected. Change metadata to plainfloat; preserve the original checkpoint and convert only that trusted locally produced scalar with an explicit safe allowlist. Do not disable safe checkpoint loading. This interruption is an implementation issue, not a benchmark failure.
