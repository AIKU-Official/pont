import gradio as gr
import torch
from PIL import Image

# test_new.py의 main 함수를 불러온다고 가정
from inference import main as test_new_main

def inference(style_image, input_sentence):
    """
    style_image: 업로드된 스타일 이미지 (PIL)
    input_sentence: 생성할 문장 (string)
    """
 
    style_path = "temp_style.png"
    style_image.save(style_path)
    
  
    class Opts:
        def __init__(self, input_sentence, style_path):
            self.cfg_file = "model/configs/Korean_scratch.yml"
            self.one_dm = "model/ckpt/final_model.pt"
            self.device = "cuda" 
            self.stable_dif_path = "runwayml/stable-diffusion-v1-5"
            self.sampling_timesteps = 50
            self.sample_method = "ddim"
            self.eta = 0.0
            self.local_rank = 0
            self.input_sentence = input_sentence
            self.style_path = style_path
            self.output_path = "output_gradio"  # 결과 저장 폴더

    opts = Opts(input_sentence, style_path)

    test_new_main(opts)


    result_path = f"{opts.output_path}/{opts.input_sentence}.png"
    result_img = Image.open(result_path).convert("RGB")

    return result_img


with gr.Blocks(
    css="""
    .gradio-container {
        text-align: center;
    }
    #main_column {
        align-items: center;
        justify-content: center;
    }
    """
) as demo:
    gr.Markdown("# 손글씨 폰트 생성하기")

    gr.Image(
        value="demo/example.png", 
        interactive=False,   
        show_label=False     
    )

    with gr.Column(elem_id="main_column"):
        gr.Markdown("한 글자 폰트 이미지를 첨부해주세요. (170x170 크기 권장)")

        style_input = gr.Image(type="pil", label="스타일 이미지 업로드")
        text_input = gr.Textbox(lines=1, placeholder="원하는 문장 입력", label="문장 입력")
        generate_btn = gr.Button("생성하기")
        result_img = gr.Image(type="pil", label="생성 결과 이미지", height=600)


    generate_btn.click(
        fn=inference,
        inputs=[style_input, text_input],
        outputs=result_img
    )

if __name__ == "__main__":
    demo.launch(share=True)
