from pyfiglet import Figlet

class LLMBannerPrinter:
    def __init__(self, word="MEDIS", font="slant"):
        self.word = word
        self.font = font
        self.figlet = Figlet(font=self.font)

    def render(self):
        return self.figlet.renderText(self.word)

    def print(self):
        print(self.render())

if __name__ == "__main__":
    banner = LLMBannerPrinter(word="//MEDIS LAB//")
    banner.print()
