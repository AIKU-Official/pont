CUDA_VISIBLE_DEVICES=0 python inference.py \
   --one_dm model/ckpt/final_model.pt \
   --input_sentence '아이쿠 한국어 손글씨 생성 프로젝트' --style_path demo/style.png --output_path output
   