Template Structure from the Paper
Core ToM Task Types
1. Unexpected Contents Task (Figure 1)
Example from paper:

"Here is a bag filled with chocolate. There is no popcorn in this bag. Yet, the label on this bag says 'popcorn' and not 'chocolate'. Sam finds the bag. Sam has never seen this bag before."

Two questions:

(a) "Sam opens the bag and looks inside. She can clearly see that it is full of ___"

Correct answer: popcorn (reality)


(b) "Sam calls a friend to tell them that she has just found a bag full of ___"

Correct answer: chocolate (false belief based on label)



Key structure:

Container labeled as S2
Actually contains S1
Protagonist encounters it for first time
Must infer: (a) reality = S1, (b) protagonist believes = S2


2. Unexpected Transfer Task
Example from paper (page 2):

"James puts his car keys in the drawer before heading out to exercise. While James is out, his wife Linda decides to clean the house. She finds the car keys in the drawer and thinks they would be safer in the key cabinet. She moves them there and continues cleaning. Later, James returns from his run and wants to get his car keys."

Two prompts:

Prompt 1: "The keys will be taken out of the key cabinet." (reality)
Prompt 2: "James will look for the keys in the drawer." (false belief)

Key structure:

Protagonist puts object in Location A
Protagonist leaves
Other agent moves object to Location B
Protagonist returns
Must infer: (a) object is in Location B, (b) protagonist believes it's in Location A


Control Conditions (page 2)
True-control task:

"If James had witnessed Linda moving the keys before leaving, the correct response to both prompts should be key cabinet instead of drawer."

This tests that models aren't just using word associations.

Template Variables (from ToM_tasks.py)
Each task has these placeholders:
python{
    'txt': '...',              # Main story text
    'o1': 'actual_content',    # S1 - what's really there
    'o2': 'false_label',       # S2 - what label/belief says
    'c': 'container',          # CX - box/bag/bottle/etc
    'xnam': 'protagonist',     # XNAM - name
    'xpro': 'he/she',         # XPRO - pronoun
    'obj_pro': 'him/her',     # OBJ_PRO - object pronoun
    'pos_pro': 'his/her',     # POS_PRO - possessive
}

Variant Types (from code patterns)
5 control variants per task:

txt (False Belief) - Standard condition

"XNAM does not open the CX and does not look inside. XNAM reads the label."


txt_correctlabel (Correct Label)

Container labeled correctly (S1 label, S1 contents)


txt_informed (Informed Protagonist)

"A cousin calls XNAM and tells OBJ_PRO that the CX has S1 in it, and that XPRO should ignore the label that says 'S2'. XNAM believes OBJ_PRO cousin."


txt_open (Open Container)

"XNAM opens the CX and looks inside. XNAM reads the label."


txt_transparent/txt_present (Transfer tasks)

Protagonist witnesses the transfer




Question Construction (from code)
Unexpected Contents:
pythonq1 = f"{xpro.capitalize()} opens the {c} and looks inside. " \
     f"{xpro.capitalize()} can clearly see that it is full of"

q2 = f"{xnam} calls a friend to tell them that {xpro} has just found " \
     f"a {c} full of"
Unexpected Transfer:
pythonq1 = "The [object] falls out of the [current_location]"
q2 = "[Protagonist] will look for the [object] in the [original_location]"

Reversal Manipulation (from code)
They also create reversed versions where S1 and S2 are swapped:
pythonif reverse:
    txt = txt.replace(o1, "####").replace(o2, o1).replace("####", o2)
This doubles the number of test items and controls for lexical biases.

Evaluation Method (page 2)

"During testing, the context is concatenated with each prompt separately to form two distinct inputs, and the LLM generates responses autoregressively for each. We evaluate the model's response by checking the first generated token using an exact match."

To pass:

First token must match expected word (e.g., "popcorn" or "chocolate")
Must get BOTH prompts correct for each story


Key Insight for Reproduction
The templates test mental state reasoning by creating scenarios where:

Reality (what's actually true) ≠ Belief (what protagonist thinks)
Public information (label/last known location) ≠ Private knowledge (what protagonist experienced)

The model must track:

Who knows what
When they learned it
What misleading cues exist