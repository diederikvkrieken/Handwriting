"""
Pre-processes an image for handwriting recognition.
Reads in a .ppm image, binarizes to use as mask, apply on original
and outputs this as .ppm
"""

import wordio, pamImage, croplib


class PreProcessor:
    """
    Class used for pre-processing
    """

    def __init__(self):
        pass

    def read(self, inputPPM):
        # Reads the input file
        self.orig = pamImage.PamImage(inputPPM)

    # Cuts regions from an image and returns these in an array
    def cut(self, inxml):
        # Read in a words xml
        lines, name = wordio.read(inxml)

        # Keep track of where we are in the file
        line_iter = iter(lines)
        cur_line = line_iter.next()
        word_iter = iter(cur_line)

        # Cut image
        crops = []  # Array of tuples (cropped images, text)
        for line in lines:
            # Iterate over lines
            for region in line:
                # Iterate over regions (words/characters)
                crops.append((croplib.crop(self.orig, region.left, region.top,
                                            region.right, region.bottom), region.text))

        # Return array
        return crops

    # def binarize(self):
    #     # Binarizes the image
    #     for pixel in self.orig:
    #         if (pixel > threshold):
    #             # One, otherwise zero

    # Ripped code from the net to write ppm
    def writeppm(self, ppm, f, ppmformat='P6'):
        assert ppmformat in ['P3', 'P6'], 'Format wrong'
        maxval = max(max(max(bit) for bit in row) for row in ppm.map)
        assert ppmformat == 'P3' or 0 <= maxval < 256, 'R,G,B must fit in a byte'
        if ppmformat == 'P6':
            fwrite = lambda s: f.write(bytes(s, 'UTF-8'))
            maxval = 255
        else:
            fwrite = f.write
            numsize=len(str(maxval))
        fwrite('%i %i\n%i\n' % (ppm.width, ppm.height, maxval))
        for h in range(ppm.height-1, -1, -1):
            for w in range(ppm.width):
                r, g, b = ppm.get(w, h)
                if ppmformat == 'P3':
                    fwrite('   %*i %*i %*i' % (numsize, r, numsize, g, numsize, b))
                else:
                    fwrite('%c%c%c' % (r, g, b))
            if ppmformat == 'P3':
                fwrite('\n')
