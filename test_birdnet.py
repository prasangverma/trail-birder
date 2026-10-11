import birdnet


def top_species(df):
    """Return (common_name, scientific_name, best_confidence) or None."""
    if df.empty:
        return None
    best = df.sort_values("confidence", ascending=False).iloc[0]
    scientific, common = best["species_name"].split("_", 1)
    return common, scientific, float(best["confidence"])


def main():
    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    predictions = model.predict("samples/koel.mp3")
    df = predictions.to_dataframe()

    print(top_species(df))
    # Count how many 3-second chunks each species appeared in
    print(df["species_name"].value_counts().head(5))


if __name__ == "__main__":
    main()
