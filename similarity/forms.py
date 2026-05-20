from django import forms

from .services.text_preprocessing import clean_text


class TextSimilarityForm(forms.Form):
    text1 = forms.CharField(
        label="Văn bản 1",
        max_length=3000,
        error_messages={
            "required": "Vui lòng nhập văn bản thứ nhất.",
            "max_length": "Văn bản thứ nhất không được vượt quá 3000 ký tự.",
        },
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 6,
            "placeholder": "Nhập văn bản thứ nhất..."
        })
    )

    text2 = forms.CharField(
        label="Văn bản 2",
        max_length=3000,
        error_messages={
            "required": "Vui lòng nhập văn bản thứ hai.",
            "max_length": "Văn bản thứ hai không được vượt quá 3000 ký tự.",
        },
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 6,
            "placeholder": "Nhập văn bản thứ hai..."
        })
    )

    def clean_text1(self):
        text = self.cleaned_data["text1"]
        if not clean_text(text):
            raise forms.ValidationError("Văn bản thứ nhất cần có nội dung có nghĩa.")
        return text

    def clean_text2(self):
        text = self.cleaned_data["text2"]
        if not clean_text(text):
            raise forms.ValidationError("Văn bản thứ hai cần có nội dung có nghĩa.")
        return text
