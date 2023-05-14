from pickletools import string1
import tkinter as tk
from flask import Flask, request, jsonify
import heapq
import math
import os
import struct
from collections import defaultdict
from fractions import Fraction
from decimal import *
import random
import string
getcontext().prec = 1000
# Huffman


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def calculate_frequency(s):
    frequency = defaultdict(int)
    for char in s:
        frequency[char] += 1
    return frequency

def create_huffman_tree(frequency):
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right

        heapq.heappush(heap, parent)

    return heap[0]

def generate_codes(node, code, codes):
    if node.char:
        codes[node.char] = code
    else:
        generate_codes(node.left, code + '0', codes)
        generate_codes(node.right, code + '1', codes)

def huffman_encode(s, codes):
    return ''.join(codes[char] for char in s)

def huffman_decode(encoded, tree):
    decoded = []
    node = tree
    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.char:
            decoded.append(node.char)
            node = tree
    return ''.join(decoded)

def huffman_compression(s):
    frequency = calculate_frequency(s)
    tree = create_huffman_tree(frequency)
    codes = {}
    generate_codes(tree, '', codes)

    encoded = huffman_encode(s, codes)
    return encoded, tree

def compressed_size_in_bits(encoded):
    return len(encoded)

# Aritmetrica

def calculate_cumulative_freq(s):
    frequency = defaultdict(int)
    for char in s:
        frequency[char] += 1
    sorted_chars = sorted(frequency.keys())
    
    cumulative_freq = {}
    total = 0
    for char in sorted_chars:
        cumulative_freq[char] = (Fraction(total), Fraction(total + frequency[char]))
        total += frequency[char]
    return cumulative_freq

def arithmetic_encode(s, cumulative_freq):
    low, high = Fraction(0), Fraction(1)

    for char in s:
        range_width = high - low
        high = low + range_width * (cumulative_freq[char][1] / len(s))
        low = low + range_width * (cumulative_freq[char][0] / len(s))

    return (low + high) / 2

def fraction_to_binary_repr(fraction, bits):
    binary_repr = []
    for _ in range(bits):
        fraction *= 2
        bit = int(fraction)
        binary_repr.append(bit)
        fraction -= bit
    return binary_repr


def arithmetic_encode(s, cumulative_freq):
    low, high = Fraction(0), Fraction(1)

    for char in s:
        range_width = high - low
        high = low + range_width * (cumulative_freq[char][1] / len(s))
        low = low + range_width * (cumulative_freq[char][0] / len(s))

    return low, high


def arithmetic_size_in_bits(low, high):
    bits = 0
    while True:
        bits += 1
        low_bin = fraction_to_binary_repr(low, bits)
        high_bin = fraction_to_binary_repr(high, bits)
        
        if low_bin[:-1] != high_bin[:-1]:
            break

    return bits


# LZW

def lzw_encode(s):
    dictionary = {chr(i): i for i in range(256)}
    current_string = ""
    encoded = []

    for char in s:
        current_string += char 
        if current_string not in dictionary:
            encoded.append(dictionary[current_string[:-1]])
            dictionary[current_string] = len(dictionary)
            current_string = char

    if current_string:
        encoded.append(dictionary[current_string])

    return encoded

def lzw_decode(encoded):
    dictionary = {i: chr(i) for i in range(256)}
    current_code = encoded.pop(0)
    decoded = [dictionary[current_code]]
    
    for code in encoded:
        if code in dictionary:
            decoded_string = dictionary[code]
        else:
            decoded_string = decoded_string + decoded_string[0]

        decoded.append(decoded_string)
        dictionary[len(dictionary)] = dictionary[current_code] + decoded_string[0]
        current_code = code

    return "".join(decoded)

def lzw_size_in_bits(encoded):
    bit_length = 9  
    dictionary_size = 512  
    bits = 0

    for code in encoded:
        bits += bit_length

        if code == dictionary_size:
            bit_length += 1
            dictionary_size *= 2

    return bits

def string_size_in_bits(s):
    return len(s) * 8
input_string = "Exemplu de compresie aritmetica"
original_size = len(input_string) * 8  # Fiecare caracter are 8 biți în format ASCII

cumulative_freq = calculate_cumulative_freq(input_string)
low, high = arithmetic_encode(input_string, cumulative_freq)
#arithmetic_decoded_string = arithmetic_decode((low + high) / 2, cumulative_freq, len(input_string))
arithmetic_encoded_size = arithmetic_size_in_bits(low, high)
compression_ratio_arithmetic = original_size / arithmetic_encoded_size



print("Dimensiune originala:", original_size, "biti")
print("Dimensiune comprimata:", arithmetic_encoded_size, "biti")
print("Rata de compresie:", compression_ratio_arithmetic)

def functie1(s):
    input_string=s
    #input_string=''.join(random.choice(string.ascii_letters + string.digits) for _ in range(1000))

    # Huffman
    huffman_encoded, huffman_tree = huffman_compression(input_string)
    huffman_decoded = huffman_decode(huffman_encoded, huffman_tree)
    original_size = string_size_in_bits(input_string)
    huffman_compressed_size = compressed_size_in_bits(huffman_encoded)
    compression_ratio_huffman = original_size / huffman_compressed_size
    if huffman_decoded == input_string:
        print("Adevarat (Huffman)")
    else:
        print("FALS (Huffman)")
    # Arithmetic
    cumulative_freq = calculate_cumulative_freq(input_string)
    arithmetic_encoded_value = arithmetic_encode(input_string, cumulative_freq)
    arithmetic_decoded_string = arithmetic_decode(arithmetic_encoded_value, cumulative_freq, len(input_string))
    arithmetic_encoded_size = arithmetic_size_in_bits(arithmetic_encoded_value)
    compression_ratio_arithmetic = original_size / arithmetic_encoded_size
    if arithmetic_decoded_string == input_string:
        print("Adevarat (Arithmetic)")
    else:
        print("FALS (Arithmetic)")

    # LZW
    lzw_encoded = lzw_encode(input_string)
    lzw_decoded = lzw_decode(lzw_encoded)
    lzw_encoded_size = lzw_size_in_bits(lzw_encoded)
    compression_ratio_lzw = original_size / lzw_encoded_size
    if lzw_decoded == input_string:
        print("Adevarat (LZW)")
    else:
        print("FALS(LZW)")
    
    
    
    print("Marime originala:", original_size)
    print("Huffman compresat marime:", huffman_compressed_size)
    print("Huffman text compresat", huffman_encoded)
    print("\n")
    print("Arithmetic compresat marime:", arithmetic_encoded_size)
    print("Arithmetic ", arithmetic_encoded_value)
    print("\n")
    print("LZW compresat marime:", lzw_encoded_size)
    print ("LZW text compresat: ",lzw_encoded)
    print("\n")

    print("arithmetic string decode",arithmetic_decoded_string)


    print("Rata de compresie (Huffman):", compression_ratio_huffman)
    print("Rata de compresie (Arithmetic):", compression_ratio_arithmetic)
    print("Rata de compresie  (LZW):", compression_ratio_lzw)

def functie(s):
    input_string = s

    # Huffman
    huffman_encoded, huffman_tree = huffman_compression(input_string)
    huffman_decoded = huffman_decode(huffman_encoded, huffman_tree)
    original_size = string_size_in_bits(input_string)
    huffman_compressed_size = compressed_size_in_bits(huffman_encoded)
    compression_ratio_huffman = original_size / huffman_compressed_size

    # Arithmetic
    cumulative_freq = calculate_cumulative_freq(input_string)
    arithmetic_encoded_value = arithmetic_encode(input_string, cumulative_freq)
    arithmetic_decoded_string = arithmetic_decode(arithmetic_encoded_value, cumulative_freq, len(input_string))
    arithmetic_encoded_size = arithmetic_size_in_bits(arithmetic_encoded_value)
    compression_ratio_arithmetic = original_size / arithmetic_encoded_size

    # LZW
    lzw_encoded = lzw_encode(input_string)
    lzw_decoded = lzw_decode(lzw_encoded)
    lzw_encoded_size = lzw_size_in_bits(lzw_encoded)
    compression_ratio_lzw = original_size / lzw_encoded_size
   

    if huffman_decoded == input_string:
        print("Adevarat (Huffman)")
        string1="Adevarat"
    else:
        string1="Fals"
    if arithmetic_decoded_string == input_string:
        string2="Adevarat"
    else:
        string2="Fals"
    if lzw_decoded == input_string:
        string3="Adevarat"
    else:
        string3="Fals"

    results = {
        "Marime originala": original_size,
        "Huffman compresat marime": huffman_compressed_size,
        "Huffman text compresat": huffman_encoded,
        "Arithmetic compresat marime": arithmetic_encoded_size,
        "Arithmetic_Nu": arithmetic_encoded_value.numerator,
        "Arithmetic_De": arithmetic_encoded_value.denominator,
        "LZW compresat marime": lzw_encoded_size,
        "LZW text compresat": lzw_encoded,
        "arithmetic string decode": arithmetic_decoded_string,
        "Rata de compresie (Huffman)": compression_ratio_huffman,
        "Rata de compresie (Arithmetic)": compression_ratio_arithmetic,
        "Rata de compresie (LZW)": compression_ratio_lzw,
        "huffCorect":string1,
        "artCorect":string2,
        "lzwCorect":string3
    }

    return results



app = Flask(__name__)

@app.route('/process_string', methods=['POST'])
def process_string():
    input_string = request.form['input_string']

    # Aici, apelați funcția care procesează string-ul și returnează rezultatele
    results = process_input(input_string)

    return jsonify(results)

def process_input(input_string):
    # Implementați logica de procesare a șirului de caractere aici
    # și returnați rezultatele într-un dicționar
    

    results = functie(input_string)
    return results

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000)