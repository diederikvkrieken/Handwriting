__author__ = 'Diederik, Diederik, Jasper, Sebastiaan, Pieter'

'''
Functions to annotate character segments based on annotations
'''

# Provides an annotation for all found segments
def annotate(self, crops, ann):
    # crops is an array of segments (images)
    # ann is a tuple of (x-start, x-end, text)

    res = []    # List of tuples (crop, annotation)
    idx = 0     # For iterating through annotations
    pos = 0     # Current x position to be considered

    # Consider all cropped characters
    for img in crops:
        width = img.shape[0]    # Get width of segment
        text = ann[idx][2]      # String found in segment, start with current annotation
        # Keep adding to text for as long as there are more segments inside
        while ann[idx][1] < pos + width:
            idx += 1            # Go to next before adding!
            text += ann[idx][2]
        if ann[idx][1] == pos + width:
            # If equal already added in loop, but not gone to next!
            idx += 1

        pos += width            # Position is now start of next character
        res.append((img, text)) # Append tuple of (character, text)
