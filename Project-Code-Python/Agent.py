# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import ImageChops, Image, ImageDraw, ImageOps
import numpy as np

rank = ["UNCHANGED", "FLIP_LEFT_RIGHT", "FLIP_TOP_BOTTOM", "ROTATE90", "ROTATE180", "ROTATE270"]
class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    
    def __init__(self):
        self.THRESHOLD = 0.03
        self.problems = {}
        self.answers = {}
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):
        if problem.problemType == "2x2":
            answer = []
            for name in problem.figures:
                if name.isdigit():
                    #self.answers[name] = Image.open(problem.figures[name].visualFilename).convert("L")
                    self.answers[name] = Image.open(problem.figures[name].visualFilename).convert("L")
                else:
                    #self.problems[name] = Image.open(problem.figures[name].visualFilename).convert("L")
                    self.problems[name] = Image.open(problem.figures[name].visualFilename).convert("L")
            trans = self.solveRPM2x2()
            if (trans[0]):
                for i in trans[0]:
                    temp = self.findSolution(self.problems['C'], i[1], self.answers, self.THRESHOLD)
                    if temp:
                        answer.append(temp)
                    
            if (trans[1]):
                for i in trans[1]:
                    temp = self.findSolution(self.problems['B'], i[1], self.answers, self.THRESHOLD)
                    if temp:
                        answer.append(temp)
            
            # check Floodfill
            if not answer:
                trans = [self.check_floodfill(self.problems['A'], self.problems['B']), self.check_floodfill(self.problems['A'], self.problems['C'])]
                if (trans[0]):
                    for i in trans[0]:
                        temp = self.findSolution(self.problems['C'], i[1], self.answers, self.THRESHOLD)
                        if temp:
                            answer.append(temp)
                    
                if (trans[1]):
                    for i in trans[1]:
                        temp = self.findSolution(self.problems['B'], i[1], self.answers, self.THRESHOLD)
                        if temp:
                            answer.append(temp)
            if answer:
                answer = answer[0][1]
            else:
                temp = sorted(self.solveSpecial())
                if temp:
                    answer = temp[0][1] 
            print(answer)
            if answer:
                return int(answer)
            return 1
        
        if problem.problemType == "3x3":
            for name in problem.figures:
                if name.isdigit():
                    self.answers[name] = Image.open(problem.figures[name].visualFilename).convert("L")
                else:
                    self.problems[name] = Image.open(problem.figures[name].visualFilename).convert("L")
            answer = self.solveRPM3X3()
            if not answer:
                return 1
            return answer
            
    def solveRPM2x2(self):
        AB = self.find_image_relationship(self.problems['A'], self.problems['B'])
        AC = self.find_image_relationship(self.problems['A'], self.problems['C'])
        return [AB, AC]
    
    def solveRPM3X3(self):
        answer = self.find_fundimental_transform()
        if answer:
            print(answer)
            return answer
        
        answer = self.addition()
        if answer:
            print(answer)
            return answer 
        answer = self.diff()
        if answer:
            print(answer)
            return answer        
        answer = self.find_xor()
        if answer:
            print(answer)
            return answer
        answer = self.chop_and_compare()
        if answer:
            print(answer)
            return answer
        

        answer = self.get_ratios()
        if answer:
            print(answer)
            return answer
        #answer = self.addition()
        print(answer)
        answer = self.find_unseen()
        
        return answer
        return 0
    
    def diff(self):
        AB = ImageChops.difference(self.problems['A'].convert("RGB"), self.problems['B'].convert("RGB")).convert("L")
        #AB.show()
        if self.unchanged(ImageOps.invert(AB), self.problems['C'], 0.05):
            for ele in self.answers:
                GH = ImageChops.difference(self.problems['G'].convert("RGB"), self.problems['H'].convert("RGB")).convert("L")
                #temp = ImageChops.add(self.problems['H'], self.answers[ele])
                if self.unchanged(ImageOps.invert(GH), self.answers[ele], 0.05):
                    return int(ele)
    
    def addition(self):
        AB = ImageChops.add(ImageOps.invert(self.problems['A']), ImageOps.invert(self.problems['B']))
        #BC = ImageChops.add(ImageOps.invert(self.problems['B']), ImageOps.invert(self.problems['C']))
        #AC = ImageChops.add(ImageOps.invert(self.problems['A']), ImageOps.invert(self.problems['C']))
        AD = ImageChops.add(ImageOps.invert(self.problems['A']), ImageOps.invert(self.problems['D']))
        #AG = ImageChops.add(ImageOps.invert(self.problems['A']), ImageOps.invert(self.problems['G']))
        if self.unchanged(ImageOps.invert(AB), self.problems['C'], 0.03):
            GH = ImageChops.add(ImageOps.invert(self.problems['G']), ImageOps.invert(self.problems['H']))
            GH = ImageOps.invert(GH)
            #GH.show()
            for ele in self.answers:
                #temp = ImageChops.add(self.problems['H'], self.answers[ele])
                if self.unchanged(GH, self.answers[ele], 0.03):
                    return int(ele)
        if self.unchanged(ImageOps.invert(AD), self.problems['G'], 0.03):
            CF = ImageChops.add(ImageOps.invert(self.problems['C']), ImageOps.invert(self.problems['F']))
            CF = ImageOps.invert(CF)
            for ele in self.answers:
                #temp = ImageChops.add(self.problems['H'], self.answers[ele])
                if self.unchanged(CF, self.answers[ele], 0.03):
                    return int(ele)
        
        AB = ImageChops.add(self.problems['A'], self.problems['B'])
        if self.unchanged(AB, self.problems['C'], 0.03):
            GH = ImageChops.add(self.problems['G'], self.problems['H'])
            #GH = ImageOps.invert(GH)
            #GH.show()
            for ele in self.answers:
                #temp = ImageChops.add(self.problems['H'], self.answers[ele])
                if self.unchanged(GH, self.answers[ele], 0.03):
                    return int(ele)
        AD = ImageChops.add(self.problems['A'], self.problems['D'])
        if self.unchanged(AD, self.problems['G'], 0.03):
            CF = ImageChops.add(self.problems['C'], self.problems['F'])
            #GH = ImageOps.invert(GH)
            #GH.show()
            for ele in self.answers:
                #temp = ImageChops.add(self.problems['H'], self.answers[ele])
                if self.unchanged(CF, self.answers[ele], 0.03):
                    return int(ele)
        
    
    # I got this idea from the Visual Reasoning Thread: finding XOR of 3 images.
    def find_xor(self):
        row1 = self.XOR3(np.array(self.problems['A']), np.array(self.problems['B']), np.array(self.problems['C']))
        row2 = self.XOR3(np.array(self.problems['D']), np.array(self.problems['E']), np.array(self.problems['F']))
        #row1.show()
        #row2.show()
        for ele in self.answers:
            row3 = self.XOR3(np.array(self.problems['G']), np.array(self.problems['H']), np.array(self.answers[ele]))
            if self.unchanged(row1, row3, 0.03) and self.unchanged(row2, row3, 0.03):
                return int(ele)
        return 0
    
    def chop_and_compare(self):
        crop_list = []
        outter_list = []
        width, height = self.problems['A'].size
        left = 0.25*width
        top = 0.25*height
        right = 0.75*width
        bottom = 0.75*height
        blank = Image.new('L', (int(width), int(height)), 255)
        for i in ['A', 'D', 'G', 'E']:
            crop_list.append(self.problems[i].crop((left,top,right,bottom)))
            #diff = ImageChops.difference(self.problems[i], temp)
            #diff.show()
            temp = self.problems[i].copy()
            white = Image.new('L', (int(width/2), int(height/2)), 255)
            temp.paste(white,(int(left), int(top)))
            #temp.show()
            outter_list.append(temp)
 #       outter_list[0].show()
 #       outter_list[1].show()
 #       outter_list[2].show()
        result = 0
        if self.unchanged(outter_list[0], outter_list[1], 0.05) and self.unchanged(outter_list[1], outter_list[2], 0.05) and not self.unchanged(outter_list[0], blank, 0.05):
            result = self.find_crop_answer()
            if result:
                return result
        if self.unchanged(outter_list[0], outter_list[3], 0.05) and not self.unchanged(outter_list[0], blank, 0.05):
            result = self.find_diagonal_crop(outter_list, crop_list)
            if result:
                return result
        return 0
    
    def find_diagonal_crop(self, outter_list, crop_list):
        width, height = self.problems['A'].size
        left = 0.25*width
        top = 0.25*height
        right = 0.75*width
        bottom = 0.75*height
        answer_list = []
        answer_crop = []
        for ele in self.answers:
            crop = self.answers[ele].crop((left,top,right,bottom))
            out = self.answers[ele].copy()
            white = Image.new('L', (int(width/2), int(height/2)), 255)
            out.paste(white,(int(left), int(top)))
            #out.show()
            if self.find_image_relationship(outter_list[0], out):
                answer_list.append(ele)
                answer_crop.append(crop)
            #print(answer_crop)
        for i in range(len(answer_crop)):
            index = True
            for j in crop_list:
                if self.unchanged(answer_crop[i], j, 0.04):
                    index = False
                    return int(answer_list[i])
            
        return 0
        
    def find_crop_answer(self):
        crop_list = []
        outter_list = []
        width, height = self.problems['A'].size
        left = 0.25*width
        top = 0.25*height
        right = 0.75*width
        bottom = 0.75*height
        for i in ['C', 'F']:
            crop_list.append(self.problems[i].crop((left,top,right,bottom)))
            #diff = ImageChops.difference(self.problems[i], temp)
            #diff.show()
            temp = self.problems[i].copy()
            white = Image.new('L', (int(width/2), int(height/2)), 255)
            temp.paste(white,(int(left), int(top)))
            #temp.show()
            outter_list.append(temp)
            
        if self.find_image_relationship(outter_list[0], outter_list[1]):
            answer_list =[]
            answer_crop = []
            for ele in self.answers:
                crop = self.answers[ele].crop((left,top,right,bottom))
                out = self.answers[ele].copy()
                white = Image.new('L', (int(width/2), int(height/2)), 255)
                out.paste(white,(int(left), int(top)))
                #out.show()
                if self.find_image_relationship(outter_list[1], out):
                    answer_list.append(ele)
                    answer_crop.append(crop)
            #print(answer_crop)
            for i in range(len(answer_crop)):
                index = True
                for j in crop_list:
                    if self.unchanged(answer_crop[i], j, 0.05):
                        index = False
                if index:
                    return int(answer_list[i])
                
        
        return 0
            
    def find_fundimental_transform(self):
        BC = self.find_image_relationship(self.problems['B'], self.problems['C'])
        DG = self.find_image_relationship(self.problems['D'], self.problems['G'])
        AC = self.unchanged(self.problems['A'], self.problems['C'], 0.05)
        AG = self.unchanged(self.problems['A'], self.problems['G'], 0.05)
        AE = self.unchanged(self.problems['A'], self.problems['E'], 0.04)
        
#        if BC:
#            temp = self.findSolution(self.problems['H'], BC[0][1], self.answers, self.THRESHOLD)
#            if temp:
#                return int(temp[1])
#        if DG:
#            temp = self.findSolution(self.problems['F'], DG[0][1], self.answers, self.THRESHOLD)
#            if temp:  
#                return int(temp[1])
        if AC and BC:
            temp = self.findSolution(self.problems['G'], "1_UNCHANGED", self.answers, self.THRESHOLD)
            if temp:  
                return int(temp[1])
        if AG and DG:
            temp = self.findSolution(self.problems['D'], "1_UNCHANGED", self.answers, self.THRESHOLD)
            if temp:  
                return int(temp[1])
        if AE:
            temp = self.findSolution(self.problems['A'], "1_UNCHANGED", self.answers, 0.05)
            if temp:  
                return int(temp[1])
        return 0  
        
    # From Visual Reasoning thread
    def XOR3(self, img1, img2, img3):
#        width, height = img1.shape
        img_sum = (255-img1) + (255-img2) + (255-img3)
        img_sum[img_sum<255] = 0
        #print(img_sum.sum())
#        for i in range(height):
#            for j in range(width):
#                unique, count =  np.unique(np.array([img1[i,j], img2[i,j], img3[i,j]]), return_counts=True)
#                print(unique)
#                if unique[0] == 0 and count == 1:
#                    total += 1  
                    
#        unique, count =  np.unique(img_sum, return_counts=True)
#        l = zip(unique, count)
#        print(list(l))
#        diff1 = ImageChops.difference(img1.convert("RGB"), img2.convert("RGB"))
#        diff2 = ImageChops.difference(diff1.convert("RGB"), img3.convert("RGB"))
        #diff3 = ImageChops.difference(img1.convert("RGB"), img3.convert("RGB"))
        #diff = ImageChops.difference(diff1.convert("RGB"), diff2.convert("RGB"))
        image = Image.fromarray(img_sum)
        #image.show()
        return image
        
    def solveSpecial(self):
        sol = []
        AB = self.find_special_relationship(self.problems['A'], self.problems['B'], self.problems['C'])
        AC = self.find_special_relationship(self.problems['A'], self.problems['C'], self.problems['B'])
        if AB:
            sol.append(AB)
        if AC:
            sol.append(AC)
        return sol
        
    def find_image_relationship(self, img1, img2):
    #reflection
        difference = 1
        result = []
        temp = self.unchanged(img1, img2, self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference,"1_UNCHANGED"])
        temp = self.unchanged(img1, img2.transpose(Image.FLIP_LEFT_RIGHT), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "2_FLIP_LEFT_RIGHT"])
        temp = self.unchanged(img1, img2.transpose(Image.FLIP_TOP_BOTTOM), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference,"3_FLIP_TOP_BOTTOM"])
        temp = self.unchanged(img1, img2.rotate(90), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "4_ROTATE90"])
        temp = self.unchanged(img1, img2.rotate(180), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "5_ROTATE180"])
        temp = self.unchanged(img1, img2.rotate(270), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "6_ROTATE270"])
        temp = self.unchanged(img1, img2.rotate(45, fillcolor="white"), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "7_ROTATE45"])
        temp = self.unchanged(img1, img2.rotate(135, fillcolor="white"), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "8_ROTATE135"])
        temp = self.unchanged(img1, img2.rotate(225, fillcolor="white"), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "9_ROTATE225"])
        temp = self.unchanged(img1, img2.rotate(315, fillcolor="white"), self.THRESHOLD)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "A_ROTATE315"])
        if result:
            return result
        return 0
    
    def find_special_relationship(self, img1, img2, img3):
        #difference
        ans = []
        img_diff = ImageChops.difference(img1.convert("RGB"), img2.convert("RGB"))
        for ele in self.answers:
            answer_diff = ImageChops.difference(img3.convert("RGB"), self.answers[ele].convert("RGB"))
            temp = self.unchanged(img_diff, answer_diff, 0.11)
            if temp:
                ans.append([temp[0], ele])
        ans.sort()
        if ans: 
            return ans[0]
        else:
            return []
            
    def check_floodfill(self, img1, img2):
        ans = []
        temp1 = self.unchanged(self.floodFill(img1), img2, self.THRESHOLD)
        temp2 = self.unchanged(img1, self.floodFill(img2), self.THRESHOLD)
        if temp1:
            ans.append([temp1[0], "B_FLOODFILL12"])
        if temp2:
            ans.append([temp2[0], "C_FLOODFILL21"])
        return ans
          
    def floodFill(self, img):
        width, height = img.size
        img_ret = img.copy()
        center = (int(0.5 * width), int(0.5 * height))
        black = 0
        ImageDraw.floodfill(img_ret, xy=center, value=black)
        return img_ret
        
    def findSolution(self, img, trans, answers, threshold):
        answer_list = []
        for ele in answers:
                if trans == "1_UNCHANGED":
                    ans = self.unchanged(img, answers[ele], threshold)
                    if ans:
                        answer_list.append([ans[0], ele])
        answer_list.sort(reverse=False)
        if answer_list:
            return answer_list[0]
        
        return []
    
    
    def unchanged(self,img1, img2, threshold):
        diff = ImageChops.difference(img1.convert("RGB"), img2.convert("RGB"))
        size = img1.size[0]*img1.size[1]
        #bbox = diff.getbbox()
        #img1.show()
        #img2.show()
        difference = self.count_nonblack_pil(diff)
        if difference/size < threshold:
            return [difference/size]
        return 0
        
#def rotation(img1, img2):
    
    def transform(self,img1, img2):
        #reflection
        difference = 1
        result = []
        temp = self.unchanged(img1, img2)
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference,"1_UNCHANGED"])
        temp = self.unchanged(img1, img2.transpose(Image.FLIP_LEFT_RIGHT))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "2_FLIP_LEFT_RIGHT"])
        temp = self.unchanged(img1, img2.transpose(Image.FLIP_TOP_BOTTOM))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference,"3_FLIP_TOP_BOTTOM"])
        temp = self.unchanged(img1, img2.rotate(90))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "4_ROTATE90"])
        temp = self.unchanged(img1, img2.rotate(180))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "5_ROTATE180"])
        temp = self.unchanged(img1, img2.rotate(270))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "6_ROTATE270"])
        temp = self.unchanged(img1, img2.rotate(45, fillcolor="white"))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "7_ROTATE45"])
        temp = self.unchanged(img1, img2.rotate(135, fillcolor="white"))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "8_ROTATE135"])
        temp = self.unchanged(img1, img2.rotate(225, fillcolor="white"))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "9_ROTATE225"])
        temp = self.unchanged(img1, img2.rotate(315, fillcolor="white"))
        if temp:
            if temp[0] < difference:
                difference = temp[0]
                result.append([difference, "A_ROTATE315"])
    
        if result:
            return sorted(result)
        return 0
    
    def compare_diff(self):
        def XOR(img1, img2):
            diff = ImageChops.difference(img1.convert("RGB"), img2.convert("RGB"))
            if self.count_nonblack_pil(img1) < self.count_nonblack_pil(img2):
                return -np.array(diff).any(axis=-1).sum()
            else:
                return np.array(diff).any(axis=-1).sum()
            
        def find_solutions(img, diff):
            for ele in self.answers:
                answer_diff = ImageChops.difference(img.convert("RGB"), self.answers[ele].convert("RGB"))
                if self.count_nonblack_pil(img) < self.count_nonblack_pil(self.answers[ele]):
                    answer_diff = -np.array(answer_diff).any(axis=-1).sum()
                else:
                    answer_diff = np.array(answer_diff).any(axis=-1).sum()
                    
                #answer_diff = sign*np.array(answer_diff).any(axis=-1).sum()
                if abs(answer_diff-diff)/(img.size[0]*img.size[1]) < self.THRESHOLD:
                    return int(ele)
            return 0
        size = self.problems['A'].size[0]*self.problems['A'].size[1]
        AB = XOR(self.problems['A'], self.problems['B'])
        BC = XOR(self.problems['B'], self.problems['C'])
        #print(AB, BC)
        if np.abs(AB-BC)/size < self.THRESHOLD:
            GH = XOR(self.problems['G'], self.problems['H'])
            temp = find_solutions(self.problems['H'], GH)
            if temp:
                return temp
        AD = XOR(self.problems['A'], self.problems['D'])
        DG = XOR(self.problems['D'], self.problems['G'])
        #print(AB, BC)
        if np.abs(AD-DG)/size < self.THRESHOLD:
            CF = XOR(self.problems['C'], self.problems['F'])
            temp = find_solutions(self.problems['F'], CF)
            if temp:
                return temp
            if self.unchanged(self.problems['F'], self.problems['H'], 0.05):
                answer = self.findSolution(self.problems['F'], "1_UNCHANGED", self.answers, 0.05)
                if answer:
                    return int(answer[1])
        return 0 
    def get_ratios(self):
        size = self.problems['A'].size[0]*self.problems['A'].size[1]
        A = size - self.count_nonblack_pil(self.problems['A'].convert("RGB"))
        B = size - self.count_nonblack_pil(self.problems['B'].convert("RGB"))
        C = size - self.count_nonblack_pil(self.problems['C'].convert("RGB"))
        if np.floor(B/A+0.5) == 2 and np.floor(C/A+0.5) == 3:
            G = size - self.count_nonblack_pil(self.problems['G'].convert("RGB"))
            for ele in self.answers:
                temp = size - self.count_nonblack_pil(self.answers[ele].convert("RGB"))
                #a = temp/G
                if np.floor(temp/G+0.5) == 3:
                    return int(ele)
                
        D = size - self.count_nonblack_pil(self.problems['D'].convert("RGB"))
        F = size - self.count_nonblack_pil(self.problems['F'].convert("RGB"))
        if np.abs(C/A-F/D) < 0.05:
            G = size - self.count_nonblack_pil(self.problems['G'].convert("RGB"))
            for ele in self.answers:
                temp = size - self.count_nonblack_pil(self.answers[ele].convert("RGB"))
                #a = temp/G
                if np.abs(temp/G-C/A) < 0.05 :
                    return int(ele)
        return 0

    def find_unseen(self):
        answers = ['1','2','3','4','5','6','7','8']
        for ele in self.answers:
                for i in self.problems:
                    #self.problems[i].show()
                    #self.answers[ele].show()
                    if self.unchanged(self.problems[i], self.answers[ele], 0.05):
                        #print("kkk", ele)
                        if ele in answers:
                            answers.remove(ele)
        if len(answers) == 1:
            return int(answers[0])
        elif len(answers) > 1:
            return int(answers[1])
        return 0
    
    # http://codereview.stackexchange.com/questions/55902/fastest-way-to-count-non-zero-pixels-using-python-and-pillow
    def count_nonblack_pil(self, img):
        """Return the number of pixels in img that are not black.
        img must be a PIL.Image object in mode RGB.

        """
        bbox = img.getbbox()
        if not bbox: return 0
        return sum(img.crop(bbox)
                   .point(lambda x: 255 if x else 0)
                   .convert("L")
                   .point(bool)
                   .getdata())
#def subtraction(img1, img2):
        
        