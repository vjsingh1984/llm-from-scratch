"""
End-to-end integration tests.

Tests the complete pipeline:
1. Tokenizer training
2. Model initialization
3. Training loop
4. Model saving/loading
5. Code generation
"""

import os
import pytest
import torch
import torch.nn.functional as F
from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import CodeTransformer
from src.model.config import CoderConfig


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_pipeline(self, sample_texts, temp_dir, device):
        """Test the complete training and generation pipeline."""

        # Step 1: Train tokenizer
        print("\n1. Training tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 200
        tokenizer.train(sample_texts, verbose=False)

        assert len(tokenizer.vocab) > 0, "Tokenizer should have vocabulary"

        # Step 2: Save tokenizer
        print("2. Saving tokenizer...")
        tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        assert os.path.exists(tokenizer_path), "Tokenizer file should exist"

        # Step 3: Create model
        print("3. Creating model...")
        config = CoderConfig(
            vocab_size=len(tokenizer.vocab),
            n_layers=2,
            d_model=64,
            n_heads=2,
            d_ff=128,
            max_seq_len=128
        )
        model = CodeTransformer(config).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Model has {num_params:,} parameters")

        # Step 4: Prepare training data
        print("4. Preparing training data...")
        train_data = []
        for text in sample_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > 10:  # Only use reasonably long sequences
                train_data.append(tokens)

        assert len(train_data) > 0, "Should have training data"

        # Step 5: Training loop (mini version)
        print("5. Training model...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        initial_loss = None
        final_loss = None

        for epoch in range(5):  # Just 5 epochs for testing
            epoch_losses = []

            for tokens in train_data:
                # Prepare batch
                if len(tokens) < 10:
                    continue

                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

                # Truncate if too long
                max_len = 50
                if input_ids.shape[1] > max_len:
                    input_ids = input_ids[:, :max_len]
                    target_ids = target_ids[:, :max_len]

                # Forward pass
                optimizer.zero_grad()
                logits, loss = model(input_ids, target_ids)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0

            if epoch == 0:
                initial_loss = avg_loss
            if epoch == 4:
                final_loss = avg_loss

            print(f"   Epoch {epoch + 1}/5: loss = {avg_loss:.4f}")

        # Loss should decrease (usually)
        if initial_loss and final_loss:
            print(f"   Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

        # Step 6: Save model
        print("6. Saving model...")
        model_path = os.path.join(temp_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, model_path)
        assert os.path.exists(model_path), "Model file should exist"

        # Step 7: Load model
        print("7. Loading model...")
        loaded_model = CodeTransformer(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()

        # Step 8: Generate code
        print("8. Generating code...")
        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(20):
                logits, _ = loaded_model(generated)
                next_logits = logits[0, -1, :] / 0.8  # temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Decode
        generated_ids = generated[0].tolist()
        generated_text = tokenizer.decode(generated_ids)

        print(f"   Generated: {generated_text[:100]}...")

        assert isinstance(generated_text, str), "Should generate text"
        assert len(generated_text) > len(prompt), "Generated text should be longer than prompt"
        assert generated_text.startswith(prompt), "Should start with prompt"

        print("\n✓ End-to-end pipeline completed successfully!")

    def test_tokenizer_persistence(self, sample_texts, temp_dir):
        """Test that tokenizer can be saved and loaded."""
        # Train
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 150
        tokenizer.train(sample_texts, verbose=False)

        # Save
        path = os.path.join(temp_dir, "tokenizer.json")
        tokenizer.save(path)

        # Load
        loaded_tokenizer = BPETokenizer()
        loaded_tokenizer.load(path)

        # Test
        for text in sample_texts:
            original = tokenizer.encode(text)
            loaded = loaded_tokenizer.encode(text)
            assert original == loaded, "Loaded tokenizer should produce same encoding"

    def test_model_persistence(self, tiny_model_config, temp_dir, device):
        """Test that model can be saved and loaded."""
        # Create model
        model = CodeTransformer(tiny_model_config).to(device)

        # Save
        path = os.path.join(temp_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': tiny_model_config,
        }, path)

        # Load
        loaded_model = CodeTransformer(tiny_model_config).to(device)
        checkpoint = torch.load(path, map_location=device)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])

        # Test
        model.eval()
        loaded_model.eval()

        test_input = torch.randint(0, tiny_model_config.vocab_size, (1, 20), device=device)

        with torch.no_grad():
            logits1, _ = model(test_input)
            logits2, _ = loaded_model(test_input)

        assert torch.allclose(logits1, logits2, rtol=1e-5), \
            "Loaded model should produce identical output"

    @pytest.mark.slow
    def test_training_improves_loss(self, sample_texts, device):
        """Test that training actually improves the model."""
        # Setup
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 200
        tokenizer.train(sample_texts, verbose=False)

        config = CoderConfig(
            vocab_size=len(tokenizer.vocab),
            n_layers=2,
            d_model=64,
            n_heads=2,
            d_ff=128
        )
        model = CodeTransformer(config).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Prepare data
        train_data = []
        for text in sample_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > 10:
                train_data.append(tokens[:50])  # Truncate

        # Measure initial loss
        initial_losses = []
        model.eval()
        with torch.no_grad():
            for tokens in train_data:
                if len(tokens) < 5:
                    continue
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
                _, loss = model(input_ids, target_ids)
                initial_losses.append(loss.item())

        initial_loss = sum(initial_losses) / len(initial_losses)

        # Train
        model.train()
        for _ in range(20):  # 20 training steps
            for tokens in train_data:
                if len(tokens) < 5:
                    continue

                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

                optimizer.zero_grad()
                _, loss = model(input_ids, target_ids)
                loss.backward()
                optimizer.step()

        # Measure final loss
        final_losses = []
        model.eval()
        with torch.no_grad():
            for tokens in train_data:
                if len(tokens) < 5:
                    continue
                input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
                _, loss = model(input_ids, target_ids)
                final_losses.append(loss.item())

        final_loss = sum(final_losses) / len(final_losses)

        print(f"\nInitial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.9, \
            f"Training should reduce loss (initial: {initial_loss:.4f}, final: {final_loss:.4f})"


@pytest.mark.integration
@pytest.mark.slow
class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_generate_multiple_scripts(self, sample_texts, device):
        """Test generating multiple scripts."""
        # Setup
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 200
        tokenizer.train(sample_texts, verbose=False)

        config = CoderConfig(
            vocab_size=len(tokenizer.vocab),
            n_layers=2,
            d_model=64,
            n_heads=2,
            d_ff=128
        )
        model = CodeTransformer(config).to(device)
        model.eval()

        # Generate from multiple prompts
        prompts = [
            "#!/bin/bash\n# Backup",
            "#!/bin/bash\n# Deploy",
            "#!/bin/bash\n# Monitor"
        ]

        results = []

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            generated = input_tensor.clone()

            with torch.no_grad():
                for _ in range(15):
                    logits, _ = model(generated)
                    next_logits = logits[0, -1, :] / 0.8
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            generated_text = tokenizer.decode(generated[0].tolist())
            results.append(generated_text)

        # Check all results
        assert len(results) == len(prompts)
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            assert result.startswith(prompt), f"Result {i} should start with prompt"
            assert len(result) > len(prompt), f"Result {i} should be longer than prompt"
