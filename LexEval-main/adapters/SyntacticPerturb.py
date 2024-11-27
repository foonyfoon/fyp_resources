import random

random.seed(42)


class SyntacticPerturber:
    def __init__(self):
        # Define perturbation functions
        self.butterfinger = self.butterfinger
        self.shuffle_words = self.shuffle_words
        self.random_upper_transform = self.random_upper_transform

    # @property
    # def __dict__(self):
    #     return {
    #         'butterfinger': self.butterfinger.__name__,
    #         'shuffle_words': self.shuffle_words.__name__,
    #         'random_upper_transform': self.random_upper_transform.__name__
    #     }

    def butterfinger(self, text):
        prob = 0.1
        """
        Function adopted from Alex Yorke's butter fingers, changed key 
        approximations for closert text and qwerty keyboard.
        """
        keyApprox = {}
        keyApprox["q"] = "qwasde12"
        keyApprox["w"] = "wqasde123"
        keyApprox["e"] = "ewsdr234"
        keyApprox["r"] = "edfgt435"
        keyApprox["t"] = "trfghy456"
        keyApprox["y"] = "ytghu567"
        keyApprox["u"] = "uyhji6789"
        keyApprox["i"] = "iujklop789"
        keyApprox["o"] = "oiklp890"
        keyApprox["p"] = "pol;[90-]"

        keyApprox["a"] = "aqwsdzx"
        keyApprox["s"] = "sqawzxde"
        keyApprox["d"] = "deswxcfr"
        keyApprox["f"] = "fdertcvg"
        keyApprox["g"] = "gfvbhrty"
        keyApprox["h"] = "hgbtnjyu"
        keyApprox["j"] = "jhnymku"
        keyApprox["k"] = "kjuml,il"
        keyApprox["l"] = "lik,.;op"

        keyApprox["z"] = "zasxcd"
        keyApprox["x"] = "xazcsd"
        keyApprox["c"] = "cxsvdf"
        keyApprox["v"] = "vcdbfg"
        keyApprox["b"] = "bvfgnh"
        keyApprox["n"] = "ngbmhj"
        keyApprox["m"] = "mnhjk,"
        keyApprox[" "] = " "

        probOfTypoArray = []
        probOfTypo = int(prob * 100)

        buttertext = ""
        for letter in text:
            lcletter = letter.lower()
            if not lcletter in keyApprox.keys():
                newletter = lcletter
            else:
                if random.choice(range(0, 100)) <= probOfTypo:
                    newletter = random.choice(keyApprox[lcletter])
                else:
                    newletter = lcletter
            # go back to original case
            if not lcletter == letter:
                newletter = newletter.upper()
            buttertext += newletter

        return buttertext

    def shuffle_words(self, text):
        words = text.split()  # Split the text into words
        random.shuffle(words)  # Shuffle the words
        perturbed_text = " ".join(
            words
        )  # Join the shuffled words back into a text
        return perturbed_text

    def random_upper_transform(self, text):
        perturbed_text = "".join(
            [
                char.upper() if random.choice([True, False]) else char
                for char in text
            ]
        )
        return perturbed_text

    def syn_perturb(
        self,
        text,
        butterfinger=None,
        shuffle_words=None,
        random_upper_transform=None,
    ):
        if butterfinger is not None:
            text = butterfinger(text)
        if shuffle_words is not None:
            text = shuffle_words(text)
        if random_upper_transform is not None:
            text = random_upper_transform(text)

        return text