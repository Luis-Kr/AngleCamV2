import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import our modules using pathlib
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from anglecam.models.base import DINOv2_AngleCam

class TestDINOv2AngleCam:
    """Test suite for DINOv2_AngleCam model."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return DictConfig({
            'model': {
                'name': 'dinov2_vits14',
                'trainable_transformer_blocks': [10, 11],
                'backbone': {
                    'hidden_dim': 384
                },
                'head': {
                    'hidden_dims': [128],
                    'dropout': 0.4
                },
                'output': {
                    'num_bins': 43
                }
            }
        })
    
    @pytest.fixture
    def mock_dinov2_backbone(self):
        """Create a mock DINOv2 backbone to avoid downloading the actual model."""
        class MockDINOv2(nn.Module):
            def __init__(self):
                super().__init__()
                # Simulate DINOv2 structure with transformer blocks
                self.blocks = nn.ModuleList([
                    nn.Linear(384, 384) for _ in range(12)  # 12 transformer blocks
                ])
                self.final_layer = nn.Linear(384, 384)
                
            def forward(self, x):
                # Simulate processing through transformer blocks
                batch_size = x.shape[0]
                return torch.randn(batch_size, 384)  # Return feature vectors
            
            def named_parameters(self, **kwargs):
                # Simulate named parameters with block structure
                params = []
                for i, block in enumerate(self.blocks):
                    for name, param in block.named_parameters():
                        params.append((f'blocks.{i}.{name}', param))
                for name, param in self.final_layer.named_parameters():
                    params.append((f'final_layer.{name}', param))
                return params
        
        return MockDINOv2()
    
    def test_model_initialization(self, mock_config, mock_dinov2_backbone):
        """Test that the model initializes correctly with proper configuration."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            
            # Check that config is stored
            assert model.cfg == mock_config
            
            # Check that backbone is properly assigned
            assert model.backbone is not None
            
            # Check that head is created
            assert model.head is not None
            assert isinstance(model.head, nn.Sequential)
    
    def test_head_creation(self, mock_config, mock_dinov2_backbone):
        """Test that the head is created with correct architecture."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            
            # Check head layers
            head_layers = list(model.head.children())
            
            # Should have: Linear -> Dropout -> GELU -> Linear -> Softmax
            assert len(head_layers) == 5
            assert isinstance(head_layers[0], nn.Linear)
            assert isinstance(head_layers[1], nn.Dropout)
            assert isinstance(head_layers[2], nn.GELU)
            assert isinstance(head_layers[3], nn.Linear)
            
            # Check dimensions
            assert head_layers[0].in_features == mock_config.model.backbone.hidden_dim
            assert head_layers[0].out_features == mock_config.model.head.hidden_dims[0]
            assert head_layers[3].in_features == mock_config.model.head.hidden_dims[0]
            assert head_layers[3].out_features == mock_config.model.output.num_bins
    
    def test_parameter_freezing(self, mock_config, mock_dinov2_backbone):
        """Test that backbone parameters are properly frozen/unfrozen."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            
            # Check that most backbone parameters are frozen
            frozen_count = 0
            trainable_count = 0
            trainable_blocks = mock_config.model.trainable_transformer_blocks
            
            for name, param in model.backbone.named_parameters():
                if any(f'blocks.{block_idx}' in name for block_idx in trainable_blocks):
                    assert param.requires_grad, f"Parameter {name} should be trainable"
                    trainable_count += 1
                else:
                    assert not param.requires_grad, f"Parameter {name} should be frozen"
                    frozen_count += 1
            
            # Ensure we have both frozen and trainable parameters
            assert frozen_count > 0, "Some backbone parameters should be frozen"
            assert trainable_count > 0, "Some backbone parameters should be trainable"
            
            # Check that all head parameters are trainable
            for param in model.head.parameters():
                assert param.requires_grad, "All head parameters should be trainable"
    
    def test_forward_pass(self, mock_config, mock_dinov2_backbone):
        """Test that forward pass works correctly."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            model.eval()
            
            # Test with different batch sizes
            for batch_size in [1, 4, 8]:
                # Create mock input (typical DINOv2 input size)
                x = torch.randn(batch_size, 3, 224, 224)
                
                with torch.no_grad():
                    output = model(x)
                
                # Check output shape
                expected_shape = (batch_size, mock_config.model.output.num_bins)
                assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
                
                # Check that output is a probability distribution (sums to ~1)
                output_sums = output.sum(dim=1)
                assert torch.allclose(output_sums, torch.ones_like(output_sums), atol=1e-6), \
                    "Output should be a probability distribution (sum to 1)"
                
                # Check that all values are non-negative (softmax output)
                assert torch.all(output >= 0), "All output values should be non-negative"
    
    def test_get_trainable_parameters(self, mock_config, mock_dinov2_backbone):
        """Test the _get_trainable_parameters method."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            
            trainable_params = model._get_trainable_parameters()
            
            # Should return a list of tensors
            assert isinstance(trainable_params, list)
            assert all(isinstance(p, torch.nn.Parameter) for p in trainable_params)
            
            # All returned parameters should have requires_grad=True
            assert all(p.requires_grad for p in trainable_params)
            
            # Compare with manual count
            manual_count = sum(1 for p in model.parameters() if p.requires_grad)
            assert len(trainable_params) == manual_count
    
    def test_model_modes(self, mock_config, mock_dinov2_backbone):
        """Test that the model can switch between train and eval modes."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            
            # Test train mode
            model.train()
            assert model.training
            assert model.backbone.training == False  # Backbone should remain in eval
            
            # Test eval mode
            model.eval()
            assert not model.training
            assert not model.backbone.training
    
    def test_device_transfer(self, mock_config, mock_dinov2_backbone):
        """Test that model can be moved between devices."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            
            # Test CPU (should always work)
            model = model.to('cpu')
            assert all(p.device.type == 'cpu' for p in model.parameters())
            
            # Test CUDA if available
            if torch.cuda.is_available():
                model = model.to('cuda')
                assert all(p.device.type == 'cuda' for p in model.parameters())
    
    def test_invalid_config(self):
        """Test error handling with invalid configuration."""
        # Test with missing required config fields
        invalid_configs = [
            DictConfig({}),  # Empty config
            DictConfig({'model': {}}),  # Missing required model fields
            DictConfig({'model': {'name': 'dinov2_vits14'}}),  # Missing other fields
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((KeyError, AttributeError)):
                with patch('torch.hub.load'):
                    DINOv2_AngleCam(invalid_config)
    
    def test_gradient_flow(self, mock_config, mock_dinov2_backbone):
        """Test that gradients flow correctly through trainable parameters."""
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(mock_config)
            model.train()
            
            # Create input and target
            x = torch.randn(2, 3, 224, 224, requires_grad=True)
            target = torch.randint(0, mock_config.model.output.num_bins, (2,))
            
            # Forward pass
            output = model(x)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Check that trainable parameters have gradients
            trainable_params = model._get_trainable_parameters()
            for param in trainable_params:
                assert param.grad is not None, "Trainable parameters should have gradients"
            
            # Check that frozen backbone parameters don't have gradients
            for name, param in model.backbone.named_parameters():
                if not param.requires_grad:
                    assert param.grad is None, f"Frozen parameter {name} should not have gradients"


# Additional integration test
class TestDINOv2AngleCamIntegration:
    """Integration tests for the model."""
    
    def test_realistic_training_step(self, mock_dinov2_backbone):
        """Test a realistic training step."""
        config = DictConfig({
            'model': {
                'name': 'dinov2_vits14',
                'trainable_transformer_blocks': [10, 11],
                'backbone': {'hidden_dim': 384},
                'head': {'hidden_dims': [128], 'dropout': 0.4},
                'output': {'num_bins': 43}
            }
        })
        
        with patch('torch.hub.load', return_value=mock_dinov2_backbone):
            model = DINOv2_AngleCam(config)
            optimizer = torch.optim.Adam(model._get_trainable_parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Simulate training step
            x = torch.randn(4, 3, 224, 224)
            targets = torch.randint(0, 43, (4,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            assert loss.item() > 0, "Loss should be positive"
            assert outputs.shape == (4, 43), "Output shape should match batch and num_bins"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])