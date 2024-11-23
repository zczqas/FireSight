import streamlit as st
from datetime import datetime


def quicksort_predictions(predictions, sort_key_func, low, high):
    """
    QuickSort implementation with pivot selection
    """

    def partition(low, high):
        # Choose middle element as pivot to handle already sorted arrays better
        mid = (low + high) // 2
        predictions[mid], predictions[high] = predictions[high], predictions[mid]

        pivot = sort_key_func(predictions[high])
        i = low - 1

        for j in range(low, high):
            if sort_key_func(predictions[j]) <= pivot:
                i += 1
                predictions[i], predictions[j] = predictions[j], predictions[i]

        predictions[i + 1], predictions[high] = predictions[high], predictions[i + 1]
        return i + 1

    if low < high:
        # Use insertion sort for small subarrays for better performance
        if high - low < 10:
            for i in range(low + 1, high + 1):
                key = predictions[i]
                key_value = sort_key_func(key)
                j = i - 1
                while j >= low and sort_key_func(predictions[j]) > key_value:
                    predictions[j + 1] = predictions[j]
                    j -= 1
                predictions[j + 1] = key
        else:
            pi = partition(low, high)
            quicksort_predictions(predictions, sort_key_func, low, pi - 1)
            quicksort_predictions(predictions, sort_key_func, pi + 1, high)


def sort_predictions(predictions, sort_by, reverse=False):
    """
    Sort predictions using QuickSort with optimization for small arrays
    """
    if not predictions:
        return predictions

    sort_keys = {
        "date": lambda x: x[6],  # timestamp
        "wildfire_prob": lambda x: x[4],  # wildfire probability
        "no_wildfire_prob": lambda x: x[3],  # no wildfire probability
    }

    if sort_by not in sort_keys:
        return predictions

    predictions_copy = predictions.copy()
    quicksort_predictions(
        predictions_copy, sort_keys[sort_by], 0, len(predictions_copy) - 1
    )

    return predictions_copy[::-1] if reverse else predictions_copy


def binary_search_date(predictions, search_date):
    """
    Binary search implementation for finding predictions by date
    Returns all predictions matching the search date
    """
    # First sort by date
    sorted_preds = sorted(predictions, key=lambda x: x[6].date())
    left, right = 0, len(sorted_preds) - 1

    # Find first occurrence
    first_occurrence = -1
    while left <= right:
        mid = (left + right) // 2
        mid_date = sorted_preds[mid][6].date()

        if mid_date == search_date:
            first_occurrence = mid
            right = mid - 1  # Continue searching left for first occurrence
        elif mid_date < search_date:
            left = mid + 1
        else:
            right = mid - 1

    if first_occurrence == -1:
        return []

    # Collect all predictions with matching date
    results = []
    i = first_occurrence
    while i < len(sorted_preds) and sorted_preds[i][6].date() == search_date:
        results.append(sorted_preds[i])
        i += 1

    return results


def search_predictions(predictions, search_term):
    """
    Enhanced search function using binary search for dates
    and efficient filtering for probability thresholds
    """
    if not search_term:
        return predictions

    search_term = search_term.lower()

    try:
        # Threshold search
        if search_term.startswith(">") or search_term.startswith("<"):
            threshold = float(search_term[1:])
            operator = search_term[0]

            # Sort by probability for more efficient filtering
            sorted_preds = sorted(
                predictions, key=lambda x: x[4]
            )  # Sort by wildfire_prob

            if operator == ">":
                return [pred for pred in sorted_preds if pred[4] > threshold]
            else:
                return [pred for pred in sorted_preds if pred[4] < threshold]

        # Date search using binary search
        elif "-" in search_term:
            try:
                search_date = datetime.strptime(search_term, "%Y-%m-%d").date()
                return binary_search_date(predictions, search_date)
            except ValueError:
                st.error("Invalid date format. Please use YYYY-MM-DD")
                return []

        return []
    except (ValueError, IndexError):
        st.error("Invalid search term")
        return []
