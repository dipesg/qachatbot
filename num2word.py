class NumberToWord:
    def __init__(self) -> None:
        
        self.nepali_to_english = {
                        '०': 0,
                        '१': 1,
                        '२': 2,
                        '३': 3,
                        '४': 4,
                        '५': 5,
                        '६': 6,
                        '७': 7,
                        '८': 8,
                        '९': 9
                            }
        
        self.number_list = [
        'एक', 'दुइ', 'तीन', 'चार', 'पाँच', 'छ', 'सात', 'आठ', 'नौ', 'दश',
        'एघार', 'बाह्र', 'तेर', 'चौध', 'पन्ध्र', 'सोर्ह', 'सत्र', 'अठार', 'उन्नाइस', 'बीस',
        'एक्काइस', 'बाईस', 'तेइस', 'चौबिस', 'पच्चीस', 'छब्बीस', 'सत्ताइस', 'अठ्ठाइस', 'उन्नतीस', 'तीस',
        'एक्तीस', 'बत्तीस', 'तेत्तीस', 'चौतीस', 'पैतिस', 'छत्तिस', ' सड्तीस', 'अड्तीस', 'उन्नचालीस', 'चालीस',
        'एकचालीस', 'ब्यालीस', 'त्रीचालीस', 'चौबालीस', 'पैतालीस', 'छ्यालीस', 'सड्चालीट्ठ', 'अड्चालीस', 'उन्नपचास', 'पचास',
        'एकाउन्न', 'बाउन्न', 'त्रीपन्न', 'चौबन्न', 'पचपन्न', 'छप्पन्न', 'सन्ताउन्न', 'अन्ठाउन्न', 'उनन्साठ्ठी', 'साठी',
        'एकसाठी', 'बयासठ्ठी', 'त्रीसठ्ठी', 'चौसठ्ठी', 'पैसठ्ठी', 'छ्यासठ्ठी', 'सड्सठ्ठी', 'अठ्सठ्ठी', 'उन्नसत्तरी', 'सत्तरी',
        'एकत्तर', 'बहत्तर', 'त्रीयत्तर', 'चौरत्तर', 'पचत्तर', 'छ्यात्तर', 'सत्तर', 'अठ्अार', 'उन्नअसी', 'असी',
        'एकासी', 'बयासी', 'त्रीरासी', 'चौरासी', 'पचासी', 'छ्यासी', 'सतासी', 'अठासी', 'उनानब्बे', 'नब्बे',
        'एकानब्बे', 'बयानब्बे', 'त्रीयानब्बे', 'चौरानब्बे', 'पञ्चानब्बे', 'छ्यानब्बे', 'सन्तानब्बे', 'अन्ठानब्बे', 'उनान्सय'
    ]
    
        self.main_list = {
            100000000000: 'खरव', 1000000000: 'अरव', 10000000: 'करोड', 100000: 'लाख', 1000: 'हजार', 100: 'सय'
        }
    

    def check_valid_number(self, nums):
        # use nums > sys.maxsize if error occurred for large numbers
        if nums > 9900000000000 or not isinstance(nums, int):
            raise ValueError("गणकले गन्न सक्ने भन्दा बढी नम्बर हाल्नुभो")

    def convert_nepali_to_english(self,nepali_number):
        english_number = 0
        for digit in nepali_number:
            if digit in self.nepali_to_english:
                english_number = english_number * 10 + self.nepali_to_english[digit]
        return english_number
    
    def ten(self, nums):
        if nums > 0:
            return self.number_list[nums - 1]
        if 1 <= nums <= 19:
            return self.number_list[nums - 1]
        elif nums % 10 == 0:
            return self.number_list[(nums // 10) + 8]
        else:
            return self.number_list[(nums // 10) + 8] + ' ' + self.number_list[(nums % 10) - 1]
  
    def nepali_name(self, nums, dividend):
        number = nums // dividend
        if number > 0:
            return self.number_list[number - 1] + ' ' + self.main_list[dividend]
        return ''
    
    def nepali_word(self, nums):
        eng_num = self.convert_nepali_to_english(nums)
        nums = abs(eng_num)
        self.check_valid_number(nums)
        result = ''
        for count, name in self.main_list.items():
            name_result = self.nepali_name(nums, count)
            if name_result:
                result += name_result + ' '
                nums %= count
                if nums <= 99:
                    result += self.ten(nums)
                    return result
        return result


# Let's convert the number to word
number_word = NumberToWord()
print(number_word.nepali_word('१६५३४२'))

# class NumberToWord:
#     def __init__(self):
#         self.nepali_to_english = {
#             '०': 0,
#             '१': 1,
#             '२': 2,
#             '३': 3,
#             '४': 4,
#             '५': 5,
#             '६': 6,
#             '७': 7,
#             '८': 8,
#             '९': 9
#         }

#         self.number_list = [
#             'एक', 'दुइ', 'तीन', 'चार', 'पाँच', 'छ', 'सात', 'आठ', 'नौ', 'दश',
#             'एघार', 'बाह्र', 'तेर', 'चौध', 'पन्ध्र', 'सोर्ह', 'सत्र', 'अठार', 'उन्नाइस', 'बीस',
#             'एक्काइस', 'बाईस', 'तेइस', 'चौबिस', 'पच्चीस', 'छब्बीस', 'सत्ताइस', 'अठ्ठाइस', 'उन्नतीस', 'तीस',
#             'एक्तीस', 'बत्तीस', 'तेत्तीस', 'चौतीस', 'पैतिस', 'छत्तिस', 'सड्तीस', 'अड्तीस', 'उन्नचालीस', 'चालीस',
#             'एकचालीस', 'ब्यालीस', 'त्रीचालीस', 'चौबालीस', 'पैतालीस', 'छ्यालीस', 'सड्चालीस', 'अड्चालीस', 'उन्नपचास', 'पचास',
#             'एकाउन्न', 'बाउन्न', 'त्रीपन्न', 'चौबन्न', 'पचपन्न', 'छप्पन्न', 'सन्ताउन्न', 'अन्ठाउन्न', 'उनन्साठ्ठी', 'साठी',
#             'एकसाठी', 'बयासठ्ठी', 'त्रीसठ्ठी', 'चौसठ्ठी', 'पैसठ्ठी', 'छ्यासठ्ठी', 'सड्सठ्ठी', 'अठ्सठ्ठी', 'उन्नसत्तरी', 'सत्तरी',
#             'एकत्तर', 'बहत्तर', 'त्रीयत्तर', 'चौरत्तर', 'पचत्तर', 'छ्यात्तर', 'सत्तर', 'अठ्अार', 'उन्नअसी', 'असी',
#             'एकासी', 'बयासी', 'त्रीरासी', 'चौरासी', 'पचासी', 'छ्यासी', 'सतासी', 'अठासी', 'उनानब्बे', 'नब्बे',
#             'एकानब्बे', 'बयानब्बे', 'त्रीयानब्बे', 'चौरानब्बे', 'पञ्चानब्बे', 'छ्यानब्बे', 'सन्तानब्बे', 'अन्ठानब्बे', 'उनान्सय'
#         ]

#         self.main_list = {
#             100000000000: 'खरव', 1000000000: 'अरव', 10000000: 'करोड', 100000: 'लाख', 1000: 'हजार', 100: 'सय'
#         }

#     def check_valid_number(self, nums):
#         if nums > 9900000000000 or not isinstance(nums, int):
#             raise ValueError("गणकले गन्न सक्ने भन्दा बढी नम्बर हाल्नुभो")

#     def convert_nepali_to_english(self, nepali_number):
#         english_number = 0
#         for digit in nepali_number:
#             if digit in self.nepali_to_english:
#                 english_number = english_number * 10 + self.nepali_to_english[digit]
#         return english_number

#     def ten(self, nums):
#         if nums > 0:
#             return self.number_list[nums - 1]
#         if 1 <= nums <= 19:
#             return self.number_list[nums - 1]
#         elif nums % 10 == 0:
#             return self.number_list[(nums // 10) + 8]
#         else:
#             return self.number_list[(nums // 10) + 8] + ' ' + self.number_list[(nums % 10) - 1]

#     def nepali_name(self, nums, dividend):
#         number = nums // dividend
#         if number > 0:
#             result = self.number_list[number - 1] + ' ' + self.main_list[dividend]
#             if nums % dividend == 0:
#                 return result
#             else:
#                 return result + ' ' + self.ten(nums % dividend)
#         return ''

#     def nepali_word(self, nums):
#         eng_num = self.convert_nepali_to_english(nums)
#         nums = abs(eng_num)
#         self.check_valid_number(nums)
#         result = ''
#         for count, name in self.main_list.items():
#             name_result = self.nepali_name(nums, count)
#             result += name_result
#             nums %= count
#         if eng_num < 0:
#             return 'माइनस ' + result
#         elif eng_num == 0:
#             return 'शून्य'
#         else:
#             return result

# num = '१०००००'
# converter = NumberToWord()
# output = converter.nepali_word(num)
# print(output)




