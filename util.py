import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_num', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_num in results.keys():
            for car_id in results[frame_num].keys():
                print(results[frame_num][car_id])
                if 'car' in results[frame_num][car_id].keys() and \
                   'license_plate' in results[frame_num][car_id].keys() and \
                   'text' in results[frame_num][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_num,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_num][car_id]['car']['bbox'][0],
                                                                results[frame_num][car_id]['car']['bbox'][1],
                                                                results[frame_num][car_id]['car']['bbox'][2],
                                                                results[frame_num][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_num][car_id]['license_plate']['bbox'][0],
                                                                results[frame_num][car_id]['license_plate']['bbox'][1],
                                                                results[frame_num][car_id]['license_plate']['bbox'][2],
                                                                results[frame_num][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_num][car_id]['license_plate']['bbox_score'],
                                                            results[frame_num][car_id]['license_plate']['text'],
                                                            results[frame_num][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) == 7:
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8'] or text[0] in dict_char_to_int.keys()) and\
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and\
            (((text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
                    (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
                        (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                            (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()))or\
                ((text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
                    (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
                        (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
                            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()))or\
                    ((text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
                        (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
                            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                                (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                                    (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()))):
            print("plaka:",text)
            return True

    elif len(text) == 8:
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8'] or text[0] in dict_char_to_int.keys()) and\
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and\
            (((text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
                    (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
                        (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                            (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
                                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()))or\
                ((text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
                    (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
                        (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
                            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
                                    (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()))or\
                    ((text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
                        (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
                            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                                (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                                    (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
                                        (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()))):
            print("plaka:",text)
            return True
        

        


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    if len(text) == 7:
        print("berkay")
        mapping = {0: dict_char_to_int, 1: dict_char_to_int,2: dict_int_to_char,3:dict_char_to_int ,4:dict_char_to_int,  5: dict_char_to_int, 6: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
    elif len(text) == 8:
        print("kenan")
        mapping = {0: dict_char_to_int, 1: dict_char_to_int,2: dict_int_to_char,3:dict_char_to_int ,4:dict_char_to_int,  5: dict_char_to_int, 6: dict_char_to_int, 7:dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_















def read_license_plate(license_plate_crop):
    
    detections = reader.readtext(license_plate_crop)
    
    for detection in detections:
        bbox, text, score = detection
        
        text = text.upper().replace(" ", "")
        
        if license_complies_format(text):
            return format_license(text), score
        
    
    return None, None



def get_car(license_plate,vehicle_track_ids):
    
    x1, y1, x2, y2, score, class_id = license_plate
    
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        car_x1, car_y1, car_x2, car_y2, car_id = vehicle_track_ids[j]
    
        if x1>car_x1 and y1>car_y1 and x2<car_x2 and y2<car_y2:
            car_index = j
            foundIt = True
            break    
        
    if foundIt:
        return vehicle_track_ids[car_index]

    
    return -1,-1,-1,-1,-1