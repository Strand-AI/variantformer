import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from utils.constants import IUPAC_CODES, SPECIAL_TOKENS


class BPEEncoder:
    def __init__(self):
        # Initialize the BPE tokenizer
        self.tokenizer = Tokenizer(BPE())

    def load_vocabulary(self, vocab_file):
        """Load the BPE vocabulary from a file."""
        self.tokenizer = Tokenizer.from_file(vocab_file)
        print(f"Loaded BPE vocabulary from {vocab_file}")

    def save_vocabulary(self, vocab_file):
        """Save the BPE vocabulary to a file."""
        self.tokenizer.save(vocab_file)
        print(f"Vocabulary saved to {vocab_file}")

    def get_vocab(self, sequences):
        iter = 0
        base_seq = IUPAC_CODES.keys()
        for s in base_seq:
            self.vocab[s] = iter
            self.merge_rule[s] = s
            iter += 1
        return

    def normalize(self, sequences):
        S = []
        for idx, seq in enumerate(sequences):
            seq = seq.upper()
            seq = "".join(
                [char if char in IUPAC_CODES else " " for char in seq]
            ).split()
            seq = [s for s in seq if s]  # Remove empty strings
            S.extend(seq)
        return S

    def encode_strand(self, text):
        encoded_seq = []
        all_sequences = []
        for seq in text:
            encoded = self.tokenizer.encode(seq)
            encoded_seq.extend(encoded.ids)
            all_sequences.extend(encoded.tokens)
        return encoded_seq, all_sequences

    def encode(self, sequences):
        if isinstance(sequences, str):
            text = sequences.split(",")
        else:
            text = sequences

        text_f = self.normalize([text[0]])
        text_r = self.normalize([text[1]])
        encoded_seq_f, all_sequences_f = self.encode_strand(text_f)
        encoded_seq_r, all_sequences_r = self.encode_strand(text_r)
        return encoded_seq_f, all_sequences_f, encoded_seq_r, all_sequences_r

    def encode_batch_forward(self, sequences: list[str]) -> list[list[int]]:
        """
        Batch encode multiple sequences at once (forward strand only).

        This is significantly faster than calling encode() in a loop because:
        1. Single Python->Rust boundary crossing
        2. Batch processing optimizations in the tokenizers library

        Args:
            sequences: List of DNA sequences to encode

        Returns:
            List of token ID lists, one per input sequence
        """
        if not sequences:
            return []

        # Normalize all sequences and track boundaries
        normalized_seqs = []
        seq_boundaries = []  # (start_idx, end_idx) for each original sequence

        for seq in sequences:
            seq_upper = seq.upper()
            # Split on non-IUPAC characters (same as normalize())
            parts = "".join(
                [char if char in IUPAC_CODES else " " for char in seq_upper]
            ).split()
            parts = [p for p in parts if p]

            start_idx = len(normalized_seqs)
            normalized_seqs.extend(parts)
            seq_boundaries.append((start_idx, len(normalized_seqs)))

        if not normalized_seqs:
            return [[] for _ in sequences]

        # Batch encode all normalized parts at once
        encodings = self.tokenizer.encode_batch(normalized_seqs)

        # Reconstruct token IDs for each original sequence
        results = []
        for start_idx, end_idx in seq_boundaries:
            token_ids = []
            for i in range(start_idx, end_idx):
                token_ids.extend(encodings[i].ids)
            results.append(token_ids)

        return results

    def decode(self, encoded_sequence):
        decoded = self.tokenizer.decode(encoded_sequence)
        return decoded.replace(" ", "")

    def encode_with_position(self, sequence, position):
        """
        Encodes a single sequence using the same normalization as encode(),
        and finds which BPE token covers the character at index `position`
        in the original (raw) sequence.

        Returns a dictionary containing:
          - 'encoded_ids': full list of token IDs for all subsequences
          - 'all_tokens': full list of token strings for all subsequences
          - 'offsets': list of (start, end) offsets for each token in the target subsequence
          - 'position_id': the global token ID covering `position` (across all subsequences)
          - 'position_token': the subword token covering `position`
          - 'target_subsequence': the specific subsequence containing the position
        """
        # Basic range check
        if position < 0 or position >= len(sequence):
            raise ValueError(
                f"Position {position} is out of range for the sequence of length {len(sequence)}."
            )

        # Apply the same normalization as the normalize() method
        sequence = sequence.upper()

        # Check if the character at target position is valid
        if sequence[position] not in IUPAC_CODES:
            raise ValueError(
                f"Position {position} points to invalid character '{sequence[position]}' "
                f"which is filtered out during normalization."
            )

        # Create subsequences exactly like normalize() does
        normalized_seq = "".join(
            [char if char in IUPAC_CODES else " " for char in sequence]
        )
        subsequences = normalized_seq.split()
        subsequences = [s for s in subsequences if s]  # Remove empty strings

        # Calculate how many invalid characters come before our target position
        invalid_chars_before_position = 0
        for i in range(position):
            if sequence[i] not in IUPAC_CODES:
                invalid_chars_before_position += 1

        # Adjust position to account for removed invalid characters
        adjusted_position = position - invalid_chars_before_position

        # Encode all subsequences and build full sequence
        all_encoded_ids = []
        all_tokens = []
        target_subseq = None
        target_subseq_idx = None
        position_in_subseq = None
        target_encoding = None

        # Find which subsequence contains our adjusted target position
        current_pos_in_normalized = 0
        global_token_offset = 0
        global_offset = 0
        for subseq_idx, subseq in enumerate(subsequences):
            # Encode this subsequence
            encoding = self.tokenizer.encode(subseq)
            all_encoded_ids.extend(encoding.ids)
            all_tokens.extend(encoding.tokens)

            subseq_start = current_pos_in_normalized
            subseq_end = current_pos_in_normalized + len(subseq)

            # Check if our adjusted target position falls within this subsequence
            if subseq_start <= adjusted_position < subseq_end:
                target_subseq = subseq
                target_subseq_idx = subseq_idx
                position_in_subseq = adjusted_position - subseq_start
                target_encoding = encoding
                global_offset = global_token_offset
            global_token_offset += len(encoding.ids)
            current_pos_in_normalized = subseq_end

        if target_subseq is None:
            raise ValueError(
                f"Could not locate adjusted position {adjusted_position} in any valid subsequence."
            )

        # Find which token covers the position within the target subsequence
        token_idx = None
        for i, (start, end) in enumerate(target_encoding.offsets):
            if start <= position_in_subseq < end:
                token_idx = i
                break

        if token_idx is None:
            raise ValueError(
                f"No token covers position {position_in_subseq} "
                f"in subsequence '{target_subseq}'."
            )

        # Calculate global position ID by adding tokens from previous subsequences
        global_position_id = global_offset + token_idx

        # Prepare the result
        return {
            "encoded_ids": all_encoded_ids,
            "all_tokens": all_tokens,
            "offsets": target_encoding.offsets,
            "position_id": global_position_id,
            "position_token": target_encoding.tokens[token_idx],
            "target_subsequence": target_subseq,
        }

    def train(self, sequences, N, dir_name, min_frequency=2):
        # Normalize sequences
        sequences = self.normalize(sequences)

        # Apply Byte Pair Encoding (BPE) and generate vocabulary
        print("Applying Byte Pair Encoding...")
        for n in N:
            # Prepare the trainer for BPE
            trainer = BpeTrainer(
                vocab_size=n,
                min_frequency=min_frequency,
                special_tokens=list(SPECIAL_TOKENS.values()),
            )

            # Train the tokenizer on the DNA sequences
            # Create a generator that yields lines from the corpus
            def dna_iterator():
                for text in sequences:
                    yield " ".join(text.split())

            self.tokenizer.train_from_iterator(dna_iterator(), trainer)

            # Save the vocabulary and merge rules to files
            vocab_file = os.path.join(
                dir_name, f"bpe_vocabulary_{n}_using_huggingface.json"
            )
            self.save_vocabulary(vocab_file)
