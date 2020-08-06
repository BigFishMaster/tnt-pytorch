import torch


def inference(cfg, model, infer_loader):
    file_name = cfg.TEST.OUTPUT_FILE
    fout = open(file_name, "w")
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            if i % 100 == 0:
                print("infer batch:", i)
            output = model(batch)
            pred = output.cpu().numpy()
            num = len(pred)
            for i in range(num):
                out = " ".join([str(_) for _ in pred[i]])
                fout.write(out + "\n")
                fout.flush()
    fout.close()
