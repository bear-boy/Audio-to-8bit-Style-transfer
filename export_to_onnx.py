# Export VITS model to ONNX

import torch
import os
from models import SynthesizerTrn
import utils
from text.symbols import symbols

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        help="model dir",
                        default=os.path.dirname(__file__) + "/pretrained/model.pth")
    parser.add_argument("--encoder_path",
                        type=str,
                        help="encoder model save path",
                        default=os.path.dirname(__file__) + "/pretrained/model_encoder.onnx")

    parser.add_argument("--decoder_path",
                        type=str,
                        help="decoder model save path",
                        default=os.path.dirname(__file__) + "/pretrained/model_decoder.onnx")

    args = parser.parse_args()

    hps = utils.get_hparams()
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)

    print('==> load model from: ', args.model_path)
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.eval()

    seq = torch.randint(low=0, high=len(symbols), size=(1,10), dtype=torch.long)
    seq_length = torch.Tensor([seq.size(1)]).long()
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    scales.unsqueeze(0)
    sid = torch.IntTensor([0]).long()

    input_names = ["seq", "seq_len", "scales", "sid"]
    output_names = ["internal_rep", "speaker_embedding"]
    model.forward = model.export_encoder_forward
    print("==> export to: ", args.encoder_path)
    torch.onnx.export(model=model,
                      args=(seq, seq_length, scales, sid),
                      f=args.encoder_path,
                      dynamic_axes={
                          "input": {
                              0: "batch",
                              1: "phonemes"
                          },
                          "input_lengths": {
                              0: "batch"
                          },
                          "scales": {
                              0: "batch"
                          },
                          "sid": {
                              0: "batch"
                          },
                          "z": {0: "batch", 2: "L"},
                          "g": {0: "batch"},
                      },
                      input_names=input_names, output_names=output_names,
                      opset_version=13, verbose=False)

    print("==> export to: ", args.decoder_path)
    input_names = ["internal_rep", "speaker_embedding"]
    output_names = ["audio"]
    model.forward = model.export_decoder_forward
    z = torch.randn((1,hps.model.inter_channels,105))
    g = torch.randn((1,hps.model.gin_channels,1))
    torch.onnx.export(model=model,
                      args=(z,g),
                      f=args.decoder_path,
                      dynamic_axes={
                          "z": {0: "batch", 2: "L"},
                          "g": {0: "batch"},
                          "output": {
                              0: "batch",
                              1: "audio",
                              2: "audio_length"
                          },
                      },
                      input_names=input_names, output_names=output_names,
                      opset_version=13, verbose=False)
