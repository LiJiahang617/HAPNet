# base exps:
# beit-l + layerdecay constructor training lr=2e-5
bash tools/dist_train.sh configs/baseline_mf/beit-adapter-l_sharesumconv-s_mf-480x640.py 2
# beit-b + layerdecay constructor training lr=2e-5 on NYU depth
bash tools/dist_train.sh configs/baseline_nyu/beit-adapter-b_sharesumconv-s_nyu-480x640.py 2
# ablation study exps:
# beit-b + mlphead on MFNet
bash tools/dist_train.sh configs/adapter_ablation/beit_adapter-b_allmlp_mf-480x640.py 2
# twinconvnext-l + mlphead on MFNet
bash tools/dist_train.sh configs/adapter_ablation/twinconvnext-l_allmlp_mf-480x640.py 2
# beit-b + uperhead/fcnhead on MFNet
bash tools/dist_train.sh configs/adapter_ablation/beit_adapter-b_upernet_mf-480x640.py 2
# twinconvnext-l + uperhead/fcnhead on MFNet
bash tools/dist_train.sh configs/adapter_ablation/twinconvnext-l_upernet_mf-480x640.py 2