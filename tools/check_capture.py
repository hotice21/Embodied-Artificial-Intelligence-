from PIL import Image, ImageStat


def main():
    path = "artifacts/frames/isaac_window_capture.png"
    image = Image.open(path)
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    print("size=%s mode=%s mean=%.2f extrema=%s" % (
        image.size,
        image.mode,
        stats.mean[0],
        gray.getextrema(),
    ))


if __name__ == "__main__":
    main()
