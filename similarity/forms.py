from django import forms


class TextSimilarityForm(forms.Form):
    text1 = forms.CharField(
        label="Văn bản 1",
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 5,
            "placeholder": "Nhập văn bản thứ nhất..."
        })
    )

    text2 = forms.CharField(
        label="Văn bản 2",
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 5,
            "placeholder": "Nhập văn bản thứ hai..."
        })
    )