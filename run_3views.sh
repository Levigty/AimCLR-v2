date

python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xview.yaml
python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xview.yaml

python main.py pretrain_aimclr_v2_3views --config config/three-stream/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml
python main.py linear_evaluation --config config/three-stream/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml











