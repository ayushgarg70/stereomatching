for i, d in enumerate(dl_eval):
    img_l = d["img_l"][0]  # 3, 360, 360
    img_r = d["img_r"][0]  # 3, 360, 360
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img_l.permute(1, 2, 0).numpy())
    ax[1].imshow(img_r.permute(1, 2, 0).numpy())
    # ax[0].set_title(f"{img_l.shape}, {img_l.dtype}, {img_l.min()}, {img_l.max()}")
    fig.tight_layout()
    plt.savefig(f"./0.png")

    inputs_l = feature_extractor(images=img_l, return_tensors="pt")["pixel_values"][0]
    inputs_r = feature_extractor(images=img_r, return_tensors="pt")["pixel_values"][0]
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(inputs_l.permute(1, 2, 0).numpy())
    ax[1].imshow(inputs_r.permute(1, 2, 0).numpy())
    # ax[0].set_title(f"{inputs_l.shape}, {inputs_l.dtype}, {inputs_l.min()}, {inputs_l.max()}")
    fig.tight_layout()
    plt.savefig(f"./1.png")

