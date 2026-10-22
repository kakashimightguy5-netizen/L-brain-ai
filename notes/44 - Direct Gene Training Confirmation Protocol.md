# Direct gene-code training confirmation

[[42 - Stronger Architecture and Evidence Protocol]]

The first stable gene-code run matched its teacher closely and quantized well, but depended on teacher/SVD initialization. Test this dependence explicitly on five fresh seeds742–746. Dense128hidden teacher, plainrank8/rank12factorized networks and generatedweightgene8/gene12 models all train directly on the same clean training images/labels for80minibatch epochs, Adam.001, gradientclip1, validation checkpoint every5epochs. NO pretrainedteacher targets/SVD initialization for the students. Initialize pre/postgene codes with variance scaling so their dotproduct has He-scaled weight variance; nonlinear decoder starts atzero residual. Matched plainfactors receive the SAME initial dotproduct weights. This isolates initialization/stability from teacher-transfer benefits. Perrowint8storage evaluated without test-selectedsettings. Bilinear coding remains equivalent tolowrank; report nonlinear gains only when supported by matchedresults.

The interrupted first spatial run did not actually stall at the clean guard: its clean CNNvalidationrose88.30%→97.21%, whilenoisyvalidationremained65.74%. The hypothesisthat theguardwasitscauseisnotconfirmed. Full-resolution revision is a separate architectural test, and the original diagnostic is retained.
