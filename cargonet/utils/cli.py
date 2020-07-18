def ask(message, default="y"):
    choices = "Y/n" if default.lower() in ("y", "yes") else "y/N"
    choice = input("%s (%s) " % (message, choices))
    values = ("y", "yes", "") if choices == "Y/n" else ("y", "yes")
    return choice.strip().lower() in values
